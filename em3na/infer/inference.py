import os
import sys
import tqdm
import shutil
import warnings
import argparse
import torch
import numpy as np
from collections import namedtuple

from em3na.utils.cryo_utils import parse_map, write_map, enlarge_grid, MRCObject
from em3na.infer.multi_gpu_wrapper import MultiGPUWrapper
from em3na.infer.torch_utils import get_device_names
from em3na.infer.gnn_inference_utils import (
    init_empty_collate_results,
    init_struct_from_translation,
    get_neighbour_idxs,
    argmin_random,
    get_inference_data,
    run_inference_on_data,
    collate_nn_results,
    get_final_nn_results, 
)
from em3na.infer.to_all_atom import affines_and_torsions_to_all_atom
from em3na.infer.flood_fill import flood_fill
from em3na.modules.model_x import Model

from em3na.io.seqio import read_fasta, is_na
from em3na.io.pdbio import atom27_to_atom23, chains_atom_pos_to_pdb, read_pdb
from em3na.utils.assign import assign_chains
from em3na.utils.misc_utils import pjoin, abspath, seed_everything
from em3na.utils.affine_utils import affine_from_3_points
from em3na.utils.struct import get_struct_empty_except

def pred_rmsd_to_plddt(
    pred_rmsd: np.ndarray,
    lower: float = 0.2,
    upper: float = 1.2,
):
    pred_rmsd_norm = (np.clip(pred_rmsd, lower, upper) - lower) / (upper - lower)
    plddt = 1.0 - pred_rmsd_norm
    plddt = (plddt * 10000.0).astype(np.int32) / 10000.0
    return plddt

def add_args(parser):
    num_res_per_run = 128
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", "--i", required=True, help="The path to the input map")
    parser.add_argument(
        "--struct", "--s", required=True, help="The path to the structure file"
    )
    parser.add_argument("--model-dir", required=True, help="Where the model at")
    parser.add_argument("--output-dir", default=".", help="Where to save the results")
    parser.add_argument("--device", default="cpu", help="Which device to run on")
    parser.add_argument(
        "--crop-length", type=int, default=num_res_per_run, help="How many points per batch"
    )
    parser.add_argument(
        "--repeat-per-residue",
        default=1,
        type=int,
        help="How many times to repeat per residue",
    )
    parser.add_argument("--refine", action="store_true", help="Run refinement program")
    parser.add_argument(
        "--batch-size", default=1, type=int, help="How many batches to run in parallel"
    )
    parser.add_argument("--fp16", action="store_true", help="Use fp16 in inference")
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=1.0,
        help="The voxel size that the GNN should be interpolating to."
    )
    # for sequence
    parser.add_argument(
        "--seq",
        type=str,
        help="Input sequence"
    )
    # for recycling
    parser.add_argument(
        "--recycle",
        type=int,
        default=3,
        help="Recycling rounds"
    )
    parser.add_argument(
        "--no_use_random_affine",
        action='store_true',
    )
    return parser


def infer(args):

    # Set output dir
    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = os.path.dirname(args.output_dir)

    device_names = get_device_names(args.device)
    num_devices = len(device_names)

    # Read structure
    struct = None
    if args.struct.endswith(".cif") or args.struct.endswith(".pdb"):
        print("# Reading structure from {}".format(args.struct))
        if args.no_use_random_affine:
            print("# Use affine from file")
            # Read atom pos
            atom_pos, _, _, _, _, _ = read_pdb(args.struct)

            # Get affine
            affines = affine_from_3_points(
                torch.from_numpy(atom_pos[..., 2, :]),
                torch.from_numpy(atom_pos[..., 1, :]),
                torch.from_numpy(atom_pos[..., 0, :]),
            ).numpy()
            affines[..., :, 0] = affines[..., :, 0] * (-1)
            affines[..., :, 2] = affines[..., :, 2] * (-1)

            # Get empty struct
            rigidgroups_gt_frames = affines[:, None, ...] # (n, 1, 3, 4)
            rigidgroups_gt_exists = np.ones( (len(affines), 1) ) # (n, 1)
            na_mask = np.ones( (len(affines), ) ) # (n, )

            struct = get_struct_empty_except(
                rigidgroups_gt_frames=rigidgroups_gt_frames,
                rigidgroups_gt_exists=rigidgroups_gt_exists,
                na_mask=na_mask,
            )

        else:
            print("# Use random affine")
            struct = init_struct_from_translation(args.struct, atom="CA")

        # check number of atoms
        if not len(struct.rigidgroups_gt_frames) >= 2:
            print("# Error cannot detect any nucleotides from this map", flush=True)
            exit(1)

    if struct is None:
        raise RuntimeError(f"File {args.struct} is not a supported file format.")

    if not (args.map.endswith(".mrc") or args.map.endswith(".map")):
        warnings.warn(f"The file {args.map} does not end with '.mrc' or '.map'\nPlease make sure it is an MRC file.")

    # Read map data
    grid, origin, _, voxel_size = parse_map(args.map, False, args.voxel_size)

    # Process grid (this does not change grid origin)
    maximum = np.percentile(grid[grid > 0.0], 99.999)
    grid = np.clip(grid, a_min=0.0, a_max=maximum)
    grid = grid / (maximum + 1e-6)
    grid = enlarge_grid(grid)

    grid_data = MRCObject(
        grid=grid,
        origin=origin, 
        voxel_size=voxel_size,
    )

    # Process structure data
    num_res = len(struct.rigidgroups_gt_frames)

    collated_results = init_empty_collate_results(num_res, device="cpu",)

    residues_left = num_res
    total_steps = num_res * args.repeat_per_residue
    steps_left_last = total_steps

    pbar = tqdm.tqdm(total=total_steps, file=sys.stdout, position=0, leave=True)

    # Get an initial set of pointers to neighbours for more efficient inference
    num_pred_residues = 50 if num_res > args.crop_length else num_res
    init_neighbours = get_neighbour_idxs(struct, k=num_pred_residues)

    model_class = Model
    model_args = {
        "use_checkpoint": False,
    }
    state_dict_path = args.model_dir

    with MultiGPUWrapper(model_class, model_args, state_dict_path, device_names, args.fp16) as wrapper:
        while residues_left > 0:
            idxs = argmin_random(
                collated_results["counts"], init_neighbours, args.batch_size * num_devices
            )
            data = get_inference_data(
                struct, grid_data, idxs, crop_length=args.crop_length, num_devices=num_devices,
            )
            results = run_inference_on_data(wrapper, data, fp16=args.fp16, run_iters=1)
            for device_id in range(num_devices):
                for i in range(args.batch_size):
                    collated_results, struct = collate_nn_results(
                        collated_results,
                        results[device_id],
                        data[device_id]["indices"],
                        struct,
                        offset=i * args.crop_length,
                        num_pred_residues=num_pred_residues,
                    )
            residues_left = (
                num_res
                - torch.sum(collated_results["counts"] > args.repeat_per_residue - 1).item()
            )
            steps_left = (
                total_steps
                - torch.sum(
                    collated_results["counts"].clip(0, args.repeat_per_residue)
                ).item()
            )
            pbar.update(n=int(steps_left_last - steps_left))
            steps_left_last = steps_left

    pbar.close()

    final_results = get_final_nn_results(collated_results)

    # Write predicted atom positions to file
    #for k, v in final_results.items():
    #    print(k, type(v), v.shape)

    # Dummy backbone
    # Convert affines torsions to all-atom 
    atom_pos, atom_mask = affines_and_torsions_to_all_atom(
        final_results['pred_affines'],
        final_results['pred_torsions'], 
    )

    # Tracing on backbone
    chains = flood_fill(
        atom_pos,
        np.ones( len(atom_pos) ),
        n_c_distance_threshold=3.5,  
    )

    # Split to chains
    chains_pred_affines = []
    chains_pred_torsions = []
    chains_pred_rmsd = []

    lens = [len(x) for x in chains]
    idxs = np.argsort(lens)[::-1]
    chains = [chains[i] for i in idxs]

    for k, c in enumerate(chains):
        """
        if len(chains[k]) < 3:
            continue
        """
        chains_pred_affines.append( final_results['pred_affines'][c] )
        chains_pred_torsions.append( final_results['pred_torsions'][c] )
        chains_pred_rmsd.append( final_results['pred_rmsd'][c] )


    # Use dummy "C" as res type
    chains_res_type = [
        np.array(
            [26] * len(x),
            dtype=np.int32,
        ) for x in chains_pred_affines
    ]


    # Convert using assigned sequences
    print("# All-atom conversion using assigned sequences")
    filter_chains_atom_pos = []
    filter_chains_atom_mask = []
    filter_chains_res_type = []
    filter_chains_bfactor = []
    filter_chains_pred_affines = []
    filter_chains_pred_torsions = []

    # Also keep the unfiltered
    unfilter_chains_atom_pos = []
    unfilter_chains_atom_mask = []
    unfilter_chains_res_type = []
    unfilter_chains_bfactor = []
    unfilter_chains_pred_affines = []
    unfilter_chains_pred_torsions = []

    for i in range(len(chains_pred_affines)):
        atom_pos, atom_mask = affines_and_torsions_to_all_atom(
            chains_pred_affines[i],
            chains_pred_torsions[i],
            chains_res_type[i],
        )
        # convert atom27 to atom23
        atom_pos, atom_mask = atom27_to_atom23(atom_pos, atom_mask, chains_res_type[i])

        # append unfiltered
        unfilter_chains_res_type.append( chains_res_type[i] )
        unfilter_chains_pred_affines.append( chains_pred_affines[i] )
        unfilter_chains_pred_torsions.append( chains_pred_torsions[i] )
        unfilter_chains_atom_pos.append(atom_pos)
        unfilter_chains_atom_mask.append(atom_mask)
        unfilter_chains_bfactor.append(
            pred_rmsd_to_plddt(
                np.repeat(chains_pred_rmsd[i][..., None], 23, axis=-1)
            )
        )

        # append filtered
        if len(chains_pred_affines[i]) >= 3:
            filter_chains_res_type.append( chains_res_type[i] )
            filter_chains_pred_affines.append( chains_pred_affines[i] )
            filter_chains_pred_torsions.append( chains_pred_torsions[i] )
            filter_chains_atom_pos.append(atom_pos)
            filter_chains_atom_mask.append(atom_mask)
            filter_chains_bfactor.append(
                pred_rmsd_to_plddt(
                    np.repeat(chains_pred_rmsd[i][..., None], 23, axis=-1)
                )
            )

    # Write affines
    fo_affines = pjoin(args.output_dir, "affines.npy")
    all_pred_affines = np.concatenate(filter_chains_pred_affines, axis=0)
    np.save(
        fo_affines,
        all_pred_affines, 
    )
    print("# Write pruned predicted affines to {}".format(fo_affines))

    fo_affines = pjoin(args.output_dir, "affines_unpruned.npy")
    all_pred_affines = np.concatenate(unfilter_chains_pred_affines, axis=0)
    np.save(
        fo_affines,
        all_pred_affines, 
    )
    print("# Write unpruned predicted affines to {}".format(fo_affines))


    # Write torsions
    fo_tors = pjoin(args.output_dir, "torsions.npy")
    all_pred_torsions = np.concatenate(filter_chains_pred_torsions, axis=0)
    np.save(
        fo_tors, 
        all_pred_torsions, 
    )
    print("# Write pruned predicted torsions to {}".format(fo_tors))

    fo_tors = pjoin(args.output_dir, "torsions_unpruned.npy")
    all_pred_torsions = np.concatenate(unfilter_chains_pred_torsions, axis=0)
    np.save(
        fo_tors, 
        all_pred_torsions, 
    )
    print("# Write unpruned predicted torsions to {}".format(fo_tors))




    # Write dummy structure for assignment
    fo = pjoin(args.output_dir, "backbone.cif")
    chains_atom_pos_to_pdb(
        fo,
        filter_chains_atom_pos,
        filter_chains_atom_mask,
        filter_chains_res_type, 
        chains_bfactor=filter_chains_bfactor,
    )
    print("# Write output structure to {}".format(fo))


    fo = pjoin(args.output_dir, "backbone_unpruned.cif")
    chains_atom_pos_to_pdb(
        fo,
        unfilter_chains_atom_pos,
        unfilter_chains_atom_mask,
        unfilter_chains_res_type, 
        chains_bfactor=unfilter_chains_bfactor,
    )
    print("# Write unpruned output structure to {}".format(fo))





def main(args):
    # setting seed
    seed_everything(42)
    assert args.recycle >= 1, "# Recycling rounds must be >= 1"

    output_dir = args.output_dir
    last_output_dir = None
    args.no_use_random_affine = False

    for i in range(args.recycle):
        print("# Infer {} / {}".format(i + 1, args.recycle))

        # Setting output dir
        args.output_dir = os.path.join(output_dir, f"recycle_{i}")

        # Infer
        infer(args)

        # Setting params for next round
        last_output_dir = os.path.join(output_dir, f"recycle_{i}")
        args.struct = os.path.join(last_output_dir, "backbone_unpruned.cif")
        args.no_use_random_affine = True

        if i == args.recycle - 1:
            shutil.copy(
                os.path.join(last_output_dir, "backbone.cif"),
                os.path.join(output_dir, "backbone.cif")
            )
            print("# Copy model")

            shutil.copy(
                os.path.join(last_output_dir, "torsions.npy"),
                os.path.join(output_dir, "torsions.npy")
            )
            shutil.copy(
                os.path.join(last_output_dir, "affines.npy"),
                os.path.join(output_dir, "affines.npy")
            )
            print("# Copy affines and torsions")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser).parse_args()
    main(args)
