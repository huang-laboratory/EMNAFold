import dataclasses
import pickle
import warnings
from typing import Dict, List

import numpy as np
import torch
from Bio.PDB import MMCIFParser, PDBParser

# Complete sequence of chain IDs supported by the PDB format.
PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)  # := 62.

STRUCT_KEYS = [
    "atom_positions",
    "aatype",
    "atom_mask",
    "residue_index",
    "chain_index",
    "chain_id",
    "b_factors",
    "rigidgroups_gt_frames",
    "rigidgroups_gt_exists",
    "rigidgroups_group_exists",
    "rigidgroups_group_is_ambiguous",
    "rigidgroups_alt_gt_frames",
    "torsion_angles_sin_cos",
    "alt_torsion_angles_sin_cos",
    "torsion_angles_mask",
    "na_mask", 
]


@dataclasses.dataclass(frozen=False)
class Struct:
    """Structure representation."""

    # Cartesian coordinates of atoms in angstroms.
    atom_positions: np.ndarray  # [num_res, 37, 3]

    # aa type for each residue represented as an integer between 21 and 29
    # DA DG DC DT  A  G  C  U
    # 21 22 23 24 25 26 27 28
    aatype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom.
    atom_mask: np.ndarray  # [num_res, 37]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_res]

    # 0-indexed number corresponding to the chain in the struct that this residue
    # belongs to.
    chain_index: np.ndarray  # [num_res]

    # The original Chain ID string list that the chain_indices correspond to.
    chain_id: np.ndarray  # [num_chains]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: np.ndarray  # [num_res, num_atom_type]

    # Frames corresponding to 'atom_positions'
    # represented as affines.
    rigidgroups_gt_frames: np.ndarray  # (num_res, 10, 3, 4)

    # Mask denoting whether the atom positions for
    # the given frame are available in the ground truth, e.g. if they were
    # resolved in the experiment.
    rigidgroups_gt_exists: np.ndarray  # (num_res, 10)

    # Mask denoting whether given group is in
    # principle present for given amino acid type.
    rigidgroups_group_exists: np.ndarray  # (num_res, 10)

    # Mask denoting whether frame is
    # affected by naming ambiguity.
    rigidgroups_group_is_ambiguous: np.ndarray  # (num_res, 10)

    # Frames with alternative atom renaming
    # corresponding to 'all_atom_positions' represented as affines
    rigidgroups_alt_gt_frames: np.ndarray  # (num_res, 10, 3, 4)

    # Array where the final 2 dimensions denote sin and cos respectively
    torsion_angles_sin_cos: np.ndarray  # (num_res, 10, 2)

    # Same as 'torsion_angles_sin_cos', but
    # with the angle shifted by pi for all chi angles affected by the naming
    # ambiguities.
    alt_torsion_angles_sin_cos: np.ndarray  # (num_res, 10, 2)

    # Mask for which chi angles are present.
    torsion_angles_mask: np.ndarray  # (num_res, 10)

    # Whether or not the residue is a na residue
    na_mask: np.ndarray  # (num_res,)

    keys = STRUCT_KEYS

def get_struct_empty_except(**kwargs) -> Struct:
    struct_dict = dict.fromkeys(STRUCT_KEYS)
    for (k, v) in kwargs.items():
        struct_dict[k] = v
    return Struct(**struct_dict)
