import torch
import numpy as np
from scipy.spatial import cKDTree

from em3na.na_utils.data import all_atom

def affines_and_torsions_to_all_atom(
    pred_affines,
    pred_torsions,
    aatype=None, 
):
    # to all atom
    num_res = len(pred_affines)

    # DA DC DG DU  A  C  G  U
    # 21 22 23 24 25 26 27 28

    if isinstance(aatype, np.ndarray):
        aatype = torch.from_numpy(aatype).long()

    if aatype is None:
        aatype = torch.full((num_res, ), 26).long()

    na_mask = torch.ones( (num_res, ) ).bool()

    pred_positions = pred_affines[..., :, -1]
    pred_positions = torch.from_numpy(pred_positions)

    pred_rotmats = pred_affines[..., :3, :3]
    pred_rotmats = torch.from_numpy(pred_rotmats)

    pred_torsions = torch.from_numpy(pred_torsions)

    pred_atom_positions = all_atom.to_atom37_rna_all_atom(
        pred_positions[None, ...], # (1, L, 3)
        pred_rotmats[None, ...], # (1, L, 3, 3)
        na_mask[None, ...], # (1, L)
        pred_torsions[None, ...], # (1, L, 10, 2)
        aatype[None, ...], # (1, L)
    ).cpu().numpy()[0]

    atom_mask = (np.abs(pred_atom_positions) > 1e-3).all(axis=-1)
   
    return pred_atom_positions, atom_mask


