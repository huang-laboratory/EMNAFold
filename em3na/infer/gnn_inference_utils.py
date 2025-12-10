import torch
import torch.nn.functional as F
import argparse

import numpy as np
from contextlib import nullcontext
from scipy.spatial import cKDTree

from em3na.utils.affine_utils import (
    get_affine_translation,
    get_affine_rot,
    get_affine,
    init_random_affine_from_translation,
)

from em3na.utils.pdb_utils import load_translation_from_file
from em3na.utils.struct import Struct, get_struct_empty_except

def argmin_random(
    count_tensor: torch.Tensor,
    neighbours: torch.LongTensor,
    batch_size: int = 1,
    repeat_per_residue: int = 3,
):
    # We first look at the individual counts for each residue
    counts = count_tensor.clamp(max=repeat_per_residue)
    # If the proportion of clamped counts is too high, we use the full count tensor
    if torch.sum(counts == repeat_per_residue).item() / len(counts) > 0.7:
        neighbour_counts = count_tensor
    else:
        neighbour_counts = counts[neighbours].sum(dim=-1)
    rand_idxs = torch.randperm(len(neighbour_counts))
    corr_idxs = torch.arange(len(neighbour_counts))[rand_idxs]
    random_argmin = neighbour_counts[rand_idxs].argsort()[:batch_size]
    original_argmin = corr_idxs[random_argmin]
    return original_argmin


def get_neighbour_idxs(struct, k: int, idxs=None):
    # Get an initial set of pointers to neighbours for more efficient inference
    backbone_frames = struct.rigidgroups_gt_frames[:, 0]  # (num_res, 3, 4)
    translation = get_affine_translation(backbone_frames)
    kd = cKDTree(translation)
    if idxs is None:
        _, init_neighbours = kd.query(translation, k=k)
    else:
        _, init_neighbours = kd.query(translation, k=k)
    return torch.from_numpy(init_neighbours)


def init_empty_collate_results(
    num_predicted_residues, unified_seq_len=None, device="cpu"
):
    result = {}
    result["counts"] = torch.zeros(num_predicted_residues, device=device)
    result["pred_positions"] = torch.zeros(num_predicted_residues, 3, device=device)
    result["pred_affines"] = torch.zeros(num_predicted_residues, 3, 4, device=device)
    result["pred_torsions"] = torch.zeros(
        num_predicted_residues, 10, 2, device=device
    )
    result["pred_rmsd"] = torch.zeros(num_predicted_residues, device=device)
    return result


def get_inference_data(
    struct, grid_data, idxs, crop_length=200, num_devices: int = 1,
):
    cryo_grids = torch.from_numpy(grid_data.grid[None])  # Add channel dim
    backbone_frames = struct.rigidgroups_gt_frames[:, 0]  # (num_res, 3, 4)
    translation = get_affine_translation(backbone_frames)
    picked_indices = np.arange(len(translation), dtype=int)

    batch = None
    batch_num = 1
    output_list = []
    batch_num_per_device = len(idxs) // num_devices
    for j in range(num_devices):
        if len(translation) > crop_length:
            kd = cKDTree(translation)
            _, picked_indices = kd.query(
                translation[idxs[j * batch_num_per_device: (j + 1) * batch_num_per_device]], k=crop_length
            )
            batch_num = batch_num_per_device
            batch = torch.concat(
                [torch.ones(crop_length, dtype=torch.long) * i for i in range(batch_num)],
                dim=0,
            )
    
        output_dict = {
            "affines": torch.from_numpy(backbone_frames[picked_indices]),
            "cryo_grids": cryo_grids, 

            "cryo_global_origins": torch.from_numpy(
                grid_data.origin.astype(np.float32)
            ),
            "cryo_voxel_sizes": torch.from_numpy(
                grid_data.voxel_size.astype(np.float32)
            ),
            "indices": torch.from_numpy(picked_indices),
            "na_mask": torch.from_numpy(struct.na_mask[picked_indices]),
            "num_nodes": len(picked_indices),
            "batch_num": batch_num,
            "batch": batch,
        }
        output_list.append(output_dict)
    return output_list


def update_struct_gt_frames(
    struct, update_indices: np.ndarray, update_affines: np.ndarray
):
    struct.rigidgroups_gt_frames[update_indices][:, 0] = update_affines
    return struct


def collate_nn_results(
    collated_results, results, indices, struct, num_pred_residues=50, offset=0,
):
    update_slice = np.s_[offset : num_pred_residues + offset]
    collated_results["counts"][indices[update_slice]] += 1

    # update positions
    collated_results["pred_positions"][indices[update_slice]] += results[
        "pred_positions"
    ][-1][update_slice]

    # update torsions
    collated_results["pred_torsions"][indices[update_slice]] += F.normalize(
        results["pred_torsions"][update_slice], p=2, dim=-1
    )

    curr_pos_avg = (
        collated_results["pred_positions"][indices[update_slice]]
        / collated_results["counts"][indices[update_slice]][..., None]
    )

    # update affines
    collated_results["pred_affines"][indices[update_slice]] = get_affine(
        get_affine_rot(results["pred_affines"][-1][update_slice]).cpu(), curr_pos_avg
    )

    # update pred rmsd
    collated_results["pred_rmsd"][indices[update_slice]] = results[
        "pred_rmsd"
    ][update_slice][..., 0]

    struct = update_struct_gt_frames(
        struct,
        indices[update_slice].numpy(),
        collated_results["pred_affines"][indices[update_slice]].numpy(),
    )
    return collated_results, struct


@torch.no_grad()
def run_inference_on_data(
    module,
    meta_batch_list,
    run_iters: int = 2,
    seq_attention_batch_size: int = 200,
    fp16: bool = False,
    using_cache: bool = False,
):
    meta_input_list = []
    for data in meta_batch_list:
        affines = data["affines"]
        kwargs = {
            #"positions": get_affine_translation(affines),
            "affines": affines,
            "run_iters": run_iters,
        }
        if data["batch_num"] == 1:
            kwargs["batch"] = None
            kwargs["cryo_grids"] = [data["cryo_grids"]]
            kwargs["cryo_global_origins"] = [data["cryo_global_origins"]]
            kwargs["cryo_voxel_sizes"] = [data["cryo_voxel_sizes"]]
        else:
            kwargs["batch"] = data["batch"]
            kwargs["cryo_grids"] = [
                data["cryo_grids"] for _ in range(data["batch_num"])
            ]
            kwargs["cryo_global_origins"] = [
                data["cryo_global_origins"] for _ in range(data["batch_num"])
            ]
            kwargs["cryo_voxel_sizes"] = [
                data["cryo_voxel_sizes"] for _ in range(data["batch_num"])
            ]
        meta_input_list.append(kwargs)
    result = module(meta_input_list)
    return result


def init_struct_from_translation(filename: str, atom: str = "C4'"):
    translation = load_translation_from_file(filename, atom) # (N, 3)
    rigidgroups_gt_frames = np.zeros((len(translation), 1, 3, 4), dtype=np.float32)
    rigidgroups_gt_frames[:, 0] = init_random_affine_from_translation(
        torch.from_numpy(translation), 
    ).numpy()
    rigidgroups_gt_exists = np.ones((len(translation), 1), dtype=np.float32)
    residue_mask = np.ones(len(rigidgroups_gt_exists), dtype=bool)
    na_mask = np.ones(len(rigidgroups_gt_exists), dtype=bool)

    return get_struct_empty_except(
        rigidgroups_gt_frames=rigidgroups_gt_frames[residue_mask],
        rigidgroups_gt_exists=rigidgroups_gt_exists[residue_mask],
        na_mask=na_mask, 
    )


def get_final_nn_results(collated_results):
    final_results = {}

    final_results["pred_positions"] = (
        collated_results["pred_positions"] / collated_results["counts"][..., None]
    )
    final_results["pred_torsions"] = (
        collated_results["pred_torsions"] / collated_results["counts"][..., None, None]
    )
    final_results["pred_affines"] = get_affine(
        get_affine_rot(collated_results["pred_affines"]),
        final_results["pred_positions"],
    )
    # plddt
    final_results["pred_rmsd"] = collated_results["pred_rmsd"]

    return dict([(k, v.numpy()) for (k, v) in final_results.items()])

