import argparse
import json
import os
from multiprocessing import Pool

import agama
import h5py
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde

agama.setUnits(mass=1, length=1, velocity=1)
units = agama.getUnits()


def init_worker(sim, sim_dir, snap, ghost_file):
    global pot_nbody, af, ghost_data

    pot_file = f"{sim_dir}{sim}/potentials/snap_{snap}/combined_snap_{snap}.ini"
    pot_nbody = agama.Potential(pot_file)
    af = agama.ActionFinder(pot_nbody, interp=False)

    ghost_data = h5py.File(ghost_file, "r")


def apply_barrier(Z, YY, E_min_grid):
    Z_mod = Z.copy()

    # mask invalid regions:
    # 1. below the minimum energy curve
    # 2. unbound energies (E > 0)
    invalid = (YY < E_min_grid) | (YY > 0)

    Z_mod[invalid] = 0.0

    return Z_mod


def process(Z_list, N_list, label, YY, E_min_grid, dz, pdf_store, n_gc_store, key):
    if len(Z_list) > 0:
        Z = np.mean(Z_list, axis=0)
        Zp = apply_barrier(Z, YY, E_min_grid)
        Zp /= Zp.sum() * dz

        pdf_store[label][key] = Zp
        n_gc_store[label][key] = np.mean(N_list)


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)


def serialize_store(store):
    out = {}
    for k, v in store.items():
        out[k] = {str(key): val.tolist() if hasattr(val, "tolist") else val for key, val in v.items()}
    return out


# mass and feh bins
def process_mass_feh_bin(locs, lg_mfo_lims, feh_lims, XX, YY, grid_positions, dz):
    i, j = locs

    # defined circular bounds:
    r_c = np.arange(-100, 100, 0.5)
    xyz_c = np.column_stack((r_c, r_c * 0, r_c * 0))
    vel_rot = np.sqrt(-r_c * pot_nbody.force(xyz_c)[:, 0])
    vxyz_c = np.column_stack((vel_rot * 0, vel_rot, vel_rot * 0))
    posvel_c = np.concatenate((xyz_c, vxyz_c), axis=1)
    ioms_c = af(posvel_c)
    jphi_c = ioms_c[:, 2]
    lz_c = jphi_c
    ep_c = pot_nbody.potential(xyz_c)
    ek_c = 0.5 * np.sum(np.array(vxyz_c) ** 2, axis=1)
    et_c = ek_c + ep_c

    log_mfor_min = lg_mfo_lims[i]["min"]
    log_mfor_max = lg_mfo_lims[i]["max"]

    feh_min = feh_lims[j]["min"]
    feh_max = feh_lims[j]["max"]

    Zis_list, Zid_list, Zes_list, Zed_list = [], [], [], []
    Nis_list, Nid_list, Nes_list, Ned_list = [], [], [], []

    for it_id in ghost_data.keys():
        print(f"({i},{j}) {it_id}")

        src_dat = ghost_data[it_id]["source"]

        situ_msk = src_dat["grpid"][()] == 0
        smsk = src_dat["s_flag"][()].astype(bool)

        fehs = src_dat["feh"][()]
        feh_msk = (feh_min <= fehs) & (fehs < feh_max)

        logm_tfor = src_dat["logm_tfor"][()]
        logm_tfor_msk = (log_mfor_min <= logm_tfor) & (logm_tfor < log_mfor_max)

        pxyz = np.array(src_dat["pxyz_snap600"])
        vxyz = np.array(src_dat["vxyz_snap600"])
        posvel = np.column_stack((pxyz, vxyz))

        ek = 0.5 * np.sum(vxyz**2, axis=1)
        ep = pot_nbody.potential(pxyz)
        et = ek + ep

        ioms = af(posvel)
        lz = ioms[:, 2]

        valid = np.isfinite(lz) & np.isfinite(et)

        lz = lz[valid]
        et = et[valid]
        smsk = smsk[valid]
        situ_msk = situ_msk[valid]
        feh_msk = feh_msk[valid]
        logm_tfor_msk = logm_tfor_msk[valid]

        mask_is = situ_msk & smsk & feh_msk & logm_tfor_msk
        mask_id = situ_msk & (~smsk) & feh_msk & logm_tfor_msk
        mask_es = (~situ_msk) & smsk & feh_msk & logm_tfor_msk
        mask_ed = (~situ_msk) & (~smsk) & feh_msk & logm_tfor_msk

        for mask, Zlist, Nlist in [
            (mask_is, Zis_list, Nis_list),
            (mask_id, Zid_list, Nid_list),
            (mask_es, Zes_list, Nes_list),
            (mask_ed, Zed_list, Ned_list),
        ]:
            lz_mskd = lz[mask]
            et_mskd = et[mask]

            Nlist.append(len(lz_mskd))

            if len(lz_mskd) > 5:
                kde = gaussian_kde(np.vstack([lz_mskd, et_mskd]))
                Z = kde(grid_positions).reshape(XX.shape)
                Z /= Z.sum() * dz
                Zlist.append(Z)

    emin_interp = interp1d(
        lz_c,
        et_c,
        bounds_error=False,
        fill_value=np.nan,
    )

    E_min_grid = emin_interp(XX)

    pdf_local = {"is": {}, "id": {}, "es": {}, "ed": {}}
    ngc_local = {"is": {}, "id": {}, "es": {}, "ed": {}}

    key = (i, j)

    process(Zis_list, Nis_list, "is", YY, E_min_grid, dz, pdf_local, ngc_local, key)
    process(Zid_list, Nid_list, "id", YY, E_min_grid, dz, pdf_local, ngc_local, key)
    process(Zes_list, Nes_list, "es", YY, E_min_grid, dz, pdf_local, ngc_local, key)
    process(Zed_list, Ned_list, "ed", YY, E_min_grid, dz, pdf_local, ngc_local, key)

    return pdf_local, ngc_local


# mass only bins
def process_mass_bin(loc, lg_mfo_lims, XX, YY, grid_positions, dz):
    i = loc

    # defined circular bounds:
    r_c = np.arange(-100, 100, 0.5)
    xyz_c = np.column_stack((r_c, r_c * 0, r_c * 0))
    vel_rot = np.sqrt(-r_c * pot_nbody.force(xyz_c)[:, 0])
    vxyz_c = np.column_stack((vel_rot * 0, vel_rot, vel_rot * 0))
    posvel_c = np.concatenate((xyz_c, vxyz_c), axis=1)
    ioms_c = af(posvel_c)
    jphi_c = ioms_c[:, 2]
    lz_c = jphi_c
    ep_c = pot_nbody.potential(xyz_c)
    ek_c = 0.5 * np.sum(np.array(vxyz_c) ** 2, axis=1)
    et_c = ek_c + ep_c

    log_mfor_min = lg_mfo_lims[i]["min"]
    log_mfor_max = lg_mfo_lims[i]["max"]

    Zis_list, Zid_list, Zes_list, Zed_list = [], [], [], []
    Nis_list, Nid_list, Nes_list, Ned_list = [], [], [], []

    for it_id in ghost_data.keys():
        print(f"({i}) {it_id}")

        src_dat = ghost_data[it_id]["source"]

        situ_msk = src_dat["grpid"][()] == 0
        smsk = src_dat["s_flag"][()].astype(bool)

        logm_tfor = src_dat["logm_tfor"][()]
        logm_tfor_msk = (log_mfor_min <= logm_tfor) & (logm_tfor < log_mfor_max)

        pxyz = np.array(src_dat["pxyz_snap600"])
        vxyz = np.array(src_dat["vxyz_snap600"])
        posvel = np.column_stack((pxyz, vxyz))

        ek = 0.5 * np.sum(vxyz**2, axis=1)
        ep = pot_nbody.potential(pxyz)
        et = ek + ep

        ioms = af(posvel)
        lz = ioms[:, 2]

        valid = np.isfinite(lz) & np.isfinite(et)

        lz = lz[valid]
        et = et[valid]
        smsk = smsk[valid]
        situ_msk = situ_msk[valid]
        logm_tfor_msk = logm_tfor_msk[valid]

        mask_is = situ_msk & smsk & logm_tfor_msk
        mask_id = situ_msk & (~smsk) & logm_tfor_msk
        mask_es = (~situ_msk) & smsk & logm_tfor_msk
        mask_ed = (~situ_msk) & (~smsk) & logm_tfor_msk

        for mask, Zlist, Nlist in [
            (mask_is, Zis_list, Nis_list),
            (mask_id, Zid_list, Nid_list),
            (mask_es, Zes_list, Nes_list),
            (mask_ed, Zed_list, Ned_list),
        ]:
            lz_mskd = lz[mask]
            et_mskd = et[mask]

            Nlist.append(len(lz_mskd))

            if len(lz_mskd) > 5:
                kde = gaussian_kde(np.vstack([lz_mskd, et_mskd]))
                Z = kde(grid_positions).reshape(XX.shape)
                Z /= Z.sum() * dz
                Zlist.append(Z)

    emin_interp = interp1d(
        lz_c,
        et_c,
        bounds_error=False,
        fill_value=np.nan,
    )

    E_min_grid = emin_interp(XX)

    pdf_local = {"is": {}, "id": {}, "es": {}, "ed": {}}
    ngc_local = {"is": {}, "id": {}, "es": {}, "ed": {}}

    key = i

    process(Zis_list, Nis_list, "is", YY, E_min_grid, dz, pdf_local, ngc_local, key)
    process(Zid_list, Nid_list, "id", YY, E_min_grid, dz, pdf_local, ngc_local, key)
    process(Zes_list, Nes_list, "es", YY, E_min_grid, dz, pdf_local, ngc_local, key)
    process(Zed_list, Ned_list, "ed", YY, E_min_grid, dz, pdf_local, ngc_local, key)

    return pdf_local, ngc_local


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--simulations", required=False, nargs="+", type=str, help="list of simulations"
    )
    parser.add_argument("-c", "--cores", required=False, type=int, help="number of cores to run process on")
    args = parser.parse_args()

    sim_lst = args.simulations
    if sim_lst is None:
        sim_lst = ["m12b", "m12c", "m12f", "m12i", "m12m"]

    cores = args.cores
    if cores is None:
        cores = 7

    sim_dir = "/Users/z5114326/Documents/simulations/"
    snap = 600

    # define limits
    feh_lims = [
        {"min": -2.0, "max": -1.5},
        {"min": -1.5, "max": -1.0},
        {"min": -1.0, "max": -0.5},
        {"min": -0.5, "max": 0.0},
    ]

    lg_mfo_lims = [
        {"min": 4.0, "max": 4.5},
        {"min": 4.5, "max": 5.0},
        {"min": 5.0, "max": 5.5},
        {"min": 5.5, "max": 6.0},
        {"min": 6.0, "max": 7.0},
    ]

    # define grid
    lz_grid = np.linspace(-12, 12, 500) * 10**3
    et_grid = np.linspace(-4, 0, 500) * 10**5
    XX, YY = np.meshgrid(lz_grid, et_grid)
    grid_positions = np.vstack([XX.ravel(), YY.ravel()])
    dz = (lz_grid[1] - lz_grid[0]) * (et_grid[1] - et_grid[0])

    for sim in sim_lst:
        print()
        print("#######################################")
        print(sim)
        print("#######################################")
        print()

        ghost_file = f"{sim_dir}{sim}/{sim}_ghosts.hdf5"

        print("Mass and Feh Bins")
        # mass and feh bins
        cache_mass_feh_pdf = "data/sim_data/" + sim + "/" + sim + "_mass_feh_pdf_store.json"
        cache_mass_feh_ngc = "data/sim_data/" + sim + "/" + sim + "_mass_feh_n_gc_store.json"

        if os.path.exists(cache_mass_feh_pdf) and os.path.exists(cache_mass_feh_ngc):
            print("Data already created")

        else:
            # define storage dicts
            pdf_mf_store = {"is": {}, "id": {}, "es": {}, "ed": {}}
            n_gc_mf_store = {"is": {}, "id": {}, "es": {}, "ed": {}}

            mf_jobs = [
                (
                    (i, j),
                    lg_mfo_lims,
                    feh_lims,
                    XX,
                    YY,
                    grid_positions,
                    dz,
                )
                for i in range(5)
                for j in range(4)
            ]

            with Pool(
                processes=min(cores, len(mf_jobs)),
                initializer=init_worker,
                initargs=(sim, sim_dir, snap, ghost_file),
            ) as pool:
                results = pool.starmap(process_mass_feh_bin, mf_jobs)

            for pdf_local, ngc_local in results:
                for grp in ["is", "id", "es", "ed"]:
                    pdf_mf_store[grp].update(pdf_local[grp])
                    n_gc_mf_store[grp].update(ngc_local[grp])

            # save only after successful computation
            save_json(serialize_store(pdf_mf_store), cache_mass_feh_pdf)
            save_json(serialize_store(n_gc_mf_store), cache_mass_feh_ngc)

            print("Results saved to JSON.")

        print("Mass Only Bins")

        # mass only bins
        cache_mass_pdf = "data/sim_data/" + sim + "/" + sim + "_mass_pdf_store.json"
        cache_mass_ngc = "data/sim_data/" + sim + "/" + sim + "_mass_n_gc_store.json"

        if os.path.exists(cache_mass_pdf) and os.path.exists(cache_mass_ngc):
            print("Data already created")

        else:
            # define storage dicts
            pdf_m_store = {"is": {}, "id": {}, "es": {}, "ed": {}}
            n_gc_m_store = {"is": {}, "id": {}, "es": {}, "ed": {}}

            m_jobs = [
                (
                    i,
                    lg_mfo_lims,
                    XX,
                    YY,
                    grid_positions,
                    dz,
                )
                for i in range(5)
            ]

            with Pool(
                processes=min(cores, len(m_jobs)),
                initializer=init_worker,
                initargs=(sim, sim_dir, snap, ghost_file),
            ) as pool:
                results = pool.starmap(process_mass_bin, m_jobs)

            for pdf_local, ngc_local in results:
                for grp in ["is", "id", "es", "ed"]:
                    pdf_m_store[grp].update(pdf_local[grp])
                    n_gc_m_store[grp].update(ngc_local[grp])

            # save only after successful computation
            save_json(serialize_store(pdf_m_store), cache_mass_pdf)
            save_json(serialize_store(n_gc_m_store), cache_mass_ngc)

            print("Results saved to JSON.")
