import argparse
import multiprocessing as mp
import os
from functools import partial

import gc_utils
import h5py
import numpy as np
import pandas as pd
import utilities as ut

_global_halt = None


def init_halt_worker(halt_obj):
    global _global_halt
    _global_halt = halt_obj


def get_halo(halt, halo_tid, snap, tid_to_index, cache):
    # Return cached result if seen before
    if (halo_tid, snap) in cache:
        return cache[(halo_tid, snap)]

    # Find index directly via prebuilt dictionary
    tidx = tid_to_index[halo_tid]
    snap_hold = halt["snapshot"][tidx]

    # Follow descendant chain until reaching or exceeding the target snapshot
    while snap_hold < snap:
        tidx = halt["descendant.index"][tidx]
        snap_hold = halt["snapshot"][tidx]

    tid = halt["tid"][tidx]
    cache[(halo_tid, snap)] = (tid, tidx)  # store result for reuse
    return tid, tidx


def weighted_avg(arr, mask, weights):
    num = np.nansum(arr * weights * mask, axis=1)
    den = np.nansum(weights * mask, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        avg = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    return np.where(np.any(mask, axis=1) & (den > 0), avg, -1)


def init_conversion(it, all_snaps, proc_file, data_dict={}):
    global _global_halt
    halt = _global_halt  # safely read it

    proc_data = h5py.File(proc_file, "r")  # open processed data file

    # conversion dictionaries
    src_vars = {
        "gc_id": "gcid",
        "group_id": "grpid",
        "feh": "feh",
        "form_time": "tfor",
        "logm_tform": "logm_tfor",
        "logm_z0": "logm_tz0",
        "ptype": "ptype",
        "pubsnap_zform": "snap_tforp",
        "t_acc": "tacc",
        "t_dis": "tdis",
        "survive_flag": "s_flag",
        "survived_accretion": "sa_flag",
        "snap_acc": "snap_tacc",
        "halo_zform": "halo_tfor",
        "snap_zform": "snap_tfor",
    }

    snp_vars = {
        "gc_id": "gcid",
        "group_id": "grpid",
        "ptype": "ptype",
        "now_accreted": "nacc_flag",
        "ecc": "ecc",
        "ek": "ek",
        "ep_agama": "ep",
        "et": "et",
        "inc": "inc",
        "lz_norm": "circ",
        "mass": "logm",
        "vel.sph": "host.vel.sph",
        "pos.sph": "host.pos.sph",
        "survive_flag": "s_flag",
        "survived_accretion": "sa_flag",
        "snap_part_idx": "pidx",
        "bound_flag": "bnd_flag",
    }

    it_id = gc_utils.iteration_name(it)
    it_dict = {"source": {}, "snapshots": {}}
    print(it_id)

    # add already known source information
    src_dat = proc_data[it_id]["source"]
    ana_msk = src_dat["analyse_flag"][()] == 1
    for var in src_vars.keys():
        it_dict["source"][src_vars[var]] = src_dat[var][ana_msk]

    # --- Precompute useful mappings ---
    src_gcid_tfor = {gcid: tfor for gcid, tfor in zip(it_dict["source"]["gcid"], it_dict["source"]["tfor"])}
    src_gcid_halo = {
        gcid: halo_tfor for gcid, halo_tfor in zip(it_dict["source"]["gcid"], it_dict["source"]["halo_tfor"])
    }

    # Prebuild a fast lookup for halo_tid → index
    tid_to_index = {tid: idx for idx, tid in enumerate(halt["tid"])}

    # Cache repeated (halo_tid, snap) results
    halo_cache = {}

    # --- Main loop ---
    for snap_id in proc_data[it_id]["snapshots"].keys():
        snap = int(snap_id[4:])
        tim = all_snaps["time_Gyr"][snap]

        snp_dat = proc_data[it_id]["snapshots"][snap_id]
        it_dict["snapshots"][snap_id] = {}

        # Copy over snapshot variables
        for var in snp_vars.keys():
            if var in snp_dat.keys():
                it_dict["snapshots"][snap_id][snp_vars[var]] = snp_dat[var][()]

        # Compute difference of tidal eigenvalues
        it_dict["snapshots"][snap_id]["tideig"] = snp_dat["tideig_1"][()] - snp_dat["tideig_3"][()]

        # Compute GC ages and host halo properties
        gcids = it_dict["snapshots"][snap_id]["gcid"]
        ages = []
        halo_tids = []
        halo_tidxs = []

        for gcid in gcids:
            ages.append(tim - src_gcid_tfor[gcid])

            halo_tid = src_gcid_halo[gcid]
            tid, tidx = get_halo(halt, halo_tid, snap, tid_to_index, halo_cache)
            halo_tids.append(tid)
            halo_tidxs.append(tidx)

        it_dict["snapshots"][snap_id]["age"] = np.array(ages)
        it_dict["snapshots"][snap_id]["halo_tid"] = np.array(halo_tids)
        it_dict["snapshots"][snap_id]["halo_tidx"] = np.array(halo_tidxs)

    data_dict[it_id] = it_dict
    proc_data.close()
    return data_dict


def environment_props(snap, it_lst, fir_dir, data_dict):
    global _global_halt
    halt = _global_halt  # safely read it

    snap_id = gc_utils.snapshot_name(snap)
    part = gc_utils.open_snapshot(snap, fir_dir, species=["dark", "star"], assign_hosts_rotation=False)

    snap_result = {}
    for it in it_lst:
        it_id = gc_utils.iteration_name(it)
        it_dict = data_dict[it_id]
        snap_result[it_id] = {"snapshots": {}}  # <-- add snapshots key
        snap_result[it_id]["snapshots"][snap_id] = {}

        snp_gcid_map = {gcid: idx for idx, gcid in enumerate(it_dict["snapshots"][snap_id]["gcid"][()])}
        halo_pos = []
        halo_vel = []

        for gcid in it_dict["snapshots"][snap_id]["gcid"][()]:
            idx = snp_gcid_map[gcid]
            pidx = it_dict["snapshots"][snap_id]["pidx"][idx]
            ptype = it_dict["snapshots"][snap_id]["ptype"][idx].decode("utf-8")
            halo_tidx = it_dict["snapshots"][snap_id]["halo_tidx"][idx]
            nacc = it_dict["snapshots"][snap_id]["nacc_flag"][idx]

            if nacc == 1:
                halo_pos.append(it_dict["snapshots"][snap_id]["host.pos.sph"][idx])
                halo_vel.append(it_dict["snapshots"][snap_id]["host.vel.sph"][idx])
            else:
                pos_xyz = ut.coordinate.get_distances(
                    part[ptype]["position"][pidx],
                    halt["position"][halo_tidx],
                    part.info["box.length"],
                    part.snapshot["scalefactor"],
                    False,
                )
                pos_sph = ut.coordinate.get_positions_in_coordinate_system(pos_xyz, "cartesian", "spherical")
                halo_pos.append(pos_sph)

                vel_xyz = ut.coordinate.get_velocity_differences(
                    part[ptype]["velocity"][pidx],
                    halt["velocity"][halo_tidx],
                    part[ptype]["position"][pidx],
                    halt["position"][halo_tidx],
                    part.info["box.length"],
                    part.snapshot["scalefactor"],
                    part.snapshot["time.hubble"],
                    False,
                )
                vel_sph = ut.coordinate.get_velocities_in_coordinate_system(
                    vel_xyz, pos_xyz, "cartesian", "spherical"
                )
                halo_vel.append(vel_sph)

        snap_result[it_id]["snapshots"][snap_id]["halo.pos.sph"] = np.array(halo_pos)
        snap_result[it_id]["snapshots"][snap_id]["halo.vel.sph"] = np.array(halo_vel)

    del part
    return snap_result


# --- Worker for computing GC averages per iteration ---
def add_averages(it, snp_lst, tim_lst, shared_dict):
    it_id = gc_utils.iteration_name(it)
    it_dict = shared_dict[it_id]
    print(it_id)

    # --- GC IDs and accretion data ---
    gcid_arr = np.array(it_dict["source"]["gcid"][()])
    tacc_arr = np.array(it_dict["source"]["snap_tacc"][()])
    n_gcs = len(gcid_arr)

    # --- Snapshot ordering ---
    snap_ids = sorted(it_dict["snapshots"].keys(), key=lambda x: int(x.replace("snap", "")))
    n_snaps = len(snap_ids)

    # --- Preallocate arrays ---
    halo_r = np.full((n_gcs, n_snaps), np.nan)
    host_r = np.full((n_gcs, n_snaps), np.nan)
    tide_m = np.full((n_gcs, n_snaps), np.nan)

    # --- Precompute GCID → index map per snapshot ---
    snap_gcid_index = {
        snap_id: {g: i for i, g in enumerate(it_dict["snapshots"][snap_id]["gcid"][()])}
        for snap_id in snap_ids
    }

    # --- Fill arrays ---
    for j, snap_id in enumerate(snap_ids):
        snap = it_dict["snapshots"][snap_id]
        gcids = snap["gcid"][()]
        idx_map = snap_gcid_index[snap_id]

        valid = np.isin(gcid_arr, gcids)
        valid_gcids = gcid_arr[valid]
        idxs = [idx_map[g] for g in valid_gcids]

        halo_r[valid, j] = snap["halo.pos.sph"][:, 0][idxs]
        host_r[valid, j] = snap["host.pos.sph"][:, 0][idxs]
        tide_m[valid, j] = snap["tideig"][idxs]

    # --- Masks and weights ---
    nan_mask = ~np.isnan(halo_r)
    weights = np.broadcast_to(tim_lst, halo_r.shape)

    # --- Birth radii ---
    first_valid_idx = np.where(nan_mask, np.arange(n_snaps), n_snaps).argmin(axis=1)
    has_valid = np.any(nan_mask, axis=1)
    birth_halo_radii = np.where(has_valid, halo_r[np.arange(n_gcs), first_valid_idx], -1)
    birth_host_radii = np.where(has_valid, host_r[np.arange(n_gcs), first_valid_idx], -1)

    # --- Time-weighted averages ---
    avg_tidems = weighted_avg(tide_m, nan_mask, weights)
    avg_halo_radii = weighted_avg(halo_r, nan_mask, weights)
    avg_host_radii = weighted_avg(host_r, nan_mask, weights)

    # --- Pre/Post accretion averages ---
    snp_arr = np.array(snp_lst)
    tacc_2d = tacc_arr[:, None]
    acc_mask = snp_arr[None, :] < tacc_2d

    valid_pre = acc_mask & nan_mask
    valid_pos = (~acc_mask) & nan_mask

    avg_tidems_pre = weighted_avg(tide_m, valid_pre, weights)
    avg_tidems_pos = weighted_avg(tide_m, valid_pos, weights)
    avg_halo_radii_pre = weighted_avg(halo_r, valid_pre, weights)
    avg_host_radii_pre = weighted_avg(host_r, valid_pre, weights)
    avg_host_radii_pos = weighted_avg(host_r, valid_pos, weights)

    # --- Assign to source dict ---
    src = it_dict["source"]
    src["halo.r.birth"] = birth_halo_radii
    src["host.r.birth"] = birth_host_radii

    src["tideig.avg"] = avg_tidems
    src["halo.r.avg"] = avg_halo_radii
    src["host.r.avg"] = avg_host_radii

    src["halo.r.avg.pre"] = avg_halo_radii_pre
    src["host.r.avg.pre"] = avg_host_radii_pre
    src["host.r.avg.pos"] = avg_host_radii_pos

    src["tideig.avg.pre"] = avg_tidems_pre
    src["tideig.avg.pos"] = avg_tidems_pos

    return {it_id: it_dict}


def create_hdf5(sim, sim_dir, shared_dict):
    save_file = os.path.join(sim_dir, sim, f"{sim}_ghosts.hdf5")  # save location

    # Ensure file exists
    if not os.path.exists(save_file):
        with h5py.File(save_file, "w") as f:
            pass  # just create empty file

    # Now open for writing/updating
    with h5py.File(save_file, "w") as f:
        for it_id, it_data in shared_dict.items():
            it_group = f.create_group(it_id)

            # --- Save source-level arrays ---
            source = it_data.get("source", {})
            source_group = it_group.create_group("source")
            for key, val in source.items():
                arr = np.array(val)
                source_group.create_dataset(key, data=arr, compression="gzip")

            # --- Save snapshots ---
            snapshots = it_data.get("snapshots", {})
            snap_group = it_group.create_group("snapshots")
            for snap_id, snap_data in snapshots.items():
                s_group = snap_group.create_group(snap_id)
                for key, val in snap_data.items():
                    arr = np.array(val)
                    s_group.create_dataset(key, data=arr, compression="gzip")

    print(f"Shared dictionary saved to {save_file}")


def worker_env_top(snap, it_lst, fir_dir, shared_dict):
    return environment_props(snap, it_lst, fir_dir, shared_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--simulation", required=True, type=str, help="simulation name (e.g. m12i)")
    parser.add_argument("-l", "--location", required=True, type=str, help="local, expansion or katana")
    parser.add_argument("-c", "--cores", required=False, type=int, help="number of cores to run process on")
    args = parser.parse_args()

    sim = args.simulation
    location = args.location

    if location == "local":
        sim_dir = "../../simulations/"
    elif location == "katana":
        sim_dir = "/srv/scratch/astro/z5114326/simulations/"
    elif location == "expansion":
        sim_dir = "/Volumes/Expansion/simulations/"
    else:
        raise RuntimeError("Incorrect simulation location provided. Must be local, katana or expansion")
    fir_dir = sim_dir + sim + "/" + sim + "_res7100"

    # it_low, it_up = args.iteration_low_limit, args.iteration_up_limit
    # it_lst = np.arange(it_low, it_up + 1, dtype=int)

    it_csv = sim_dir + "iteration_check.csv"
    it_df = pd.read_csv(it_csv)
    it_msk = it_df[sim] == 0
    it_id_lst = it_df["it_id"][it_msk].values
    it_lst = np.array([int(it_id[2:]) for it_id in it_id_lst])
    print(it_lst)

    cores = args.cores
    if cores is None:
        cores = 6

    halt = gc_utils.get_halo_tree(sim, sim_dir, assign_hosts_rotation=False)

    proc_file = sim_dir + sim + "/" + sim + "_processed.hdf5"

    all_data = sim_dir + sim + "/" + sim + "_res7100/snapshot_times.txt"
    all_snaps = pd.read_table(all_data, comment="#", header=None, sep=r"\s+")
    all_snaps.columns = [
        "index",
        "scale_factor",
        "redshift",
        "time_Gyr",
        "lookback_time_Gyr",
        "time_width_Myr",
    ]

    pub_data = sim_dir + "/snapshot_times_public.txt"
    pub_snaps = pd.read_table(pub_data, comment="#", header=None, sep=r"\s+")
    pub_snaps.columns = [
        "index",
        "scale_factor",
        "redshift",
        "time_Gyr",
        "lookback_time_Gyr",
        "time_width_Myr",
    ]
    snp_lst = pub_snaps["index"].values
    tim_lst = pub_snaps["time_Gyr"].values

    # --- First MP: init_conversion ---
    print(f"Running {len(it_lst)} iterations on {cores} cores...")
    with mp.Pool(processes=cores, maxtasksperchild=1, initializer=init_halt_worker, initargs=(halt,)) as pool:
        task_args = [(it, all_snaps, proc_file) for it in it_lst]
        results = pool.starmap(init_conversion, task_args, chunksize=1)

    # --- Combine results after first MP ---
    shared_dict = {k: v for d in results for k, v in d.items()}

    # --- Second MP: environment_props ---
    print(f"Computing environment properties for {len(snp_lst)} snapshots on {cores} cores...")

    # Use functools.partial to "pre-fill" extra arguments
    worker_partial = partial(worker_env_top, it_lst=it_lst, fir_dir=fir_dir, shared_dict=shared_dict)

    with mp.Pool(processes=cores, maxtasksperchild=1, initializer=init_halt_worker, initargs=(halt,)) as pool:
        results = pool.map(worker_partial, snp_lst, chunksize=1)

    # Merge results back into shared_dict
    for snap_result in results:
        for it_id, it_vals in snap_result.items():
            snap_id = list(it_vals["snapshots"].keys())[0]  # get snapshot name
            shared_dict[it_id]["snapshots"][snap_id].update(it_vals["snapshots"][snap_id])

    # --- Run the multiprocess ---
    print(f"Computing GC averages for {len(it_lst)} iterations on {cores} cores...")

    task_args = [(it, snp_lst, tim_lst, shared_dict) for it in it_lst]

    with mp.Pool(processes=cores, maxtasksperchild=1, initializer=init_halt_worker, initargs=(halt,)) as pool:
        results = pool.starmap(add_averages, task_args, chunksize=1)

    # Merge back into shared_dict
    for res in results:
        shared_dict.update(res)

    del halt

    create_hdf5(sim, sim_dir, shared_dict)
