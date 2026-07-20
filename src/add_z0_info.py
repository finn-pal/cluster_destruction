import argparse

import agama
import gc_utils
import gizmo_analysis as gizmo
import h5py
import numpy as np
from scipy.signal import find_peaks

agama.setUnits(mass=1, length=1, velocity=1)
units = agama.getUnits()

# Input Functions ###########################################################################


def remove_duplicates_with_report(arr):
    arr = np.array(arr)  # Convert the list to a numpy array
    unique_elements, counts = np.unique(arr, return_counts=True)  # Get unique elements and their counts
    duplicates = unique_elements[counts > 1]  # Duplicates are those that appear more than once
    unique_list = unique_elements.tolist()  # Convert unique elements back to a list

    return unique_list, duplicates.tolist()  # Return unique and duplicate lists


def get_snap600_data(ghost_data, part600, fir_dir, pot_nbody):
    # get correct indexes
    print("Create Index Mapping")

    it_dict = {}
    for it_id in ghost_data.keys():
        # get all gcids and corresponding ptypes
        gcids_src = ghost_data[it_id]["source"]["gcid"][()]
        ptype_byte_src = ghost_data[it_id]["source"]["ptype"][()]
        ptypes_src = [ptype.decode("utf-8") for ptype in ptype_byte_src]

        # Step 1: group GCs by particle type
        gc_by_ptype = {}
        gc_by_ptype["star"] = []
        gc_by_ptype["dark"] = []

        for gc, ptype in zip(gcids_src, ptypes_src):
            gc_by_ptype[ptype].append(gc)

        # Step 2: for each ptype, build a small dict: gc_id → index
        id_idx_map = {}
        for ptype, gcids in gc_by_ptype.items():
            ids = part600[ptype]["id"]  # all part ids

            # Check which of these are in the main list
            mask = np.isin(ids, gcids)
            idxs = np.nonzero(mask)[0]
            found_ids = ids[idxs]

            # Build small, efficient lookup: GC ID → array index
            id_idx_map[ptype] = dict(zip(found_ids, idxs))

            # concerned abour duplciate star ids
            if ptype == "star":
                _, duplicates_ids = remove_duplicates_with_report(found_ids)

        ids_set = set(duplicates_ids)

        # Build snapshot → GCID coverage map
        snap_to_ids = {}
        for snap_id in ghost_data[it_id]["snapshots"].keys():
            gcids_snap = set(ghost_data[it_id]["snapshots"][snap_id]["gcid"][()])
            common = ids_set & gcids_snap
            if len(common) > 0:
                snap_to_ids[snap_id] = common

        # Remove GCIDs that do not exist in any snapshot
        all_ids_in_snaps = set().union(*snap_to_ids.values())
        removed_ids = ids_set - all_ids_in_snaps
        ids_set -= removed_ids

        # if len(removed_ids) > 0:
        #     print("Removed GCIDs not present in any snapshot:", removed_ids)

        # Greedy minimum-snapshot assignment
        remaining_ids = ids_set.copy()
        assignment = {}

        while len(remaining_ids) > 0:
            best_snap = max(snap_to_ids, key=lambda s: len(snap_to_ids[s] & remaining_ids))

            covered = snap_to_ids[best_snap] & remaining_ids

            for gcid in covered:
                assignment[gcid] = best_snap

            remaining_ids -= covered

        # Invert assignment: snapshot → list of GCIDs
        snap_to_gcids = {}
        for gcid, snap_id in assignment.items():
            snap_to_gcids.setdefault(snap_id, []).append(gcid)

        for snap_id in snap_to_gcids:
            snap_to_gcids[snap_id].sort()

        snap_to_idxs = {}
        for snap_id, gcids in snap_to_gcids.items():
            gcid_arr = ghost_data[it_id]["snapshots"][snap_id]["gcid"][()]
            gcid_to_idx = {gcid: i for i, gcid in enumerate(gcid_arr)}

            gcidxs = sorted(gcid_to_idx[gcid] for gcid in gcids if gcid in gcid_to_idx)

            snap_to_idxs[snap_id] = ghost_data[it_id]["snapshots"][snap_id]["pidx"][gcidxs]

        it_dict[it_id] = {"id_idx_map": id_idx_map, "snap_to_idxs": snap_to_idxs}

    snap_id_lst = np.array([])
    for it_id in ghost_data.keys():
        snap_hold = np.array(list(it_dict[it_id]["snap_to_idxs"].keys()))
        snap_id_lst = np.concatenate((snap_id_lst, snap_hold))

    snap_id_lst = np.unique(snap_id_lst)

    for snap_id in snap_id_lst:
        snap_get = int(snap_id[4:])
        part_get = gizmo.io.Read.read_snapshots(["star"], "index", snap_get, fir_dir, assign_pointers=True)
        pointers = part_get.Pointer.get_pointers(
            species_name_from="star", species_names_to="star", forward=True
        )

        for it_id in ghost_data.keys():
            snap_hold = np.array(list(it_dict[it_id]["snap_to_idxs"].keys()))

            if snap_id not in snap_hold:
                continue

            indices_at_snap600 = pointers[it_dict[it_id]["snap_to_idxs"][snap_id]]
            for pidx in indices_at_snap600:
                gcid = part600["star"]["id"][pidx]
                it_dict[it_id]["id_idx_map"]["star"][gcid] = pidx

    print("Index Map created")
    print("Get positions and velocities")

    prop_dict = {}
    for it_id in ghost_data.keys():
        prop_dict[it_id] = {}

        src_dat = ghost_data[it_id]["source"]
        gcids_src = src_dat["gcid"][()]
        ptype_byte_src = src_dat["ptype"][()]
        ptypes_src = [ptype.decode("utf-8") for ptype in ptype_byte_src]

        pidx_snap600 = []
        pxyz_snap600 = []
        vxyz_snap600 = []
        for gcid, ptype in zip(gcids_src, ptypes_src):
            pidx = it_dict[it_id]["id_idx_map"][ptype][gcid]
            pxyz = part600[ptype].prop("host.distance.principal", pidx)
            vxyz = part600[ptype].prop("host.velocity.principal", pidx)

            pidx_snap600.append(pidx)
            pxyz_snap600.append(pxyz)
            vxyz_snap600.append(vxyz)

        # prop_dict[it_id]["gcid"] = gcids_src
        # prop_dict[it_id]["ptype"] = ptype_byte_src
        prop_dict[it_id]["pidx_snap600"] = np.array(pidx_snap600)
        prop_dict[it_id]["pxyz_snap600"] = np.array(pxyz_snap600)
        prop_dict[it_id]["vxyz_snap600"] = np.array(vxyz_snap600)

    print("Positions and velocities got")
    print("Get orbital times")

    for it_id in ghost_data.keys():
        pxyz = prop_dict[it_id]["pxyz_snap600"]
        vxyz = prop_dict[it_id]["vxyz_snap600"]

        posvel = np.column_stack((pxyz, vxyz))
        result = agama.orbit(potential=pot_nbody, ic=posvel, time=10 * pot_nbody.Tcirc(posvel), trajsize=1000)

        t_orb = []
        for result_i in result:
            t_i_agama = result_i[0]
            t_i_myr = t_i_agama * units["time"]  # Myr
            orb_i = result_i[1]

            r_i = np.linalg.norm(orb_i[:, 0:3], axis=1)
            peaks, _ = find_peaks(r_i)

            if len(peaks) < 2:
                t_orb_i = np.nan
            else:
                t_orb_i = np.mean(np.diff(t_i_myr[peaks]))

            t_orb.append(t_orb_i)
        t_orb = np.array(t_orb)

        prop_dict[it_id]["torb_600"] = t_orb

    print("Orbital times got")

    return prop_dict


def add_to_hdf5(ghost_data, prop_dict):
    for it_id in ghost_data.keys():
        src_dat = ghost_data[it_id]["source"]
        for dataset in prop_dict[it_id].keys():
            if dataset in src_dat.keys():
                del src_dat[dataset]

            src_dat.create_dataset(dataset, data=prop_dict[it_id][dataset])


# Main Input ################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--simulation", required=True, type=str, help="simulation name (e.g. m12i)")
    parser.add_argument("-l", "--location", required=True, type=str, help="local, expansion or katana")
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

    print("Setting up")

    ghost_file = f"{sim_dir}{sim}/{sim}_ghosts.hdf5"
    ghost_data = h5py.File(ghost_file, "a")

    snap = 600
    part600 = gc_utils.open_snapshot(snap, fir_dir, ["star", "dark"])

    pot_file = sim_dir + sim + "/potentials/snap_%d/combined_snap_%d.ini" % (snap, snap)
    pot_nbody = agama.Potential(pot_file)

    print("Set up complete")
    print("Get data")

    prop_dict = get_snap600_data(ghost_data, part600, fir_dir, pot_nbody)

    print("Data got")
    print("Save to file")

    add_to_hdf5(ghost_data, prop_dict)

    ghost_data.close()

    print("All done :)")
