import os
import sys
import argparse
from functools import reduce

import numpy as np
from em3na.utils.grid import get_aa_type
from em3na.io.seqio import read_fasta, get_seq_align_among_candidates
from em3na.io.pdbio import read_pdb, chains_atom_pos_to_pdb, split_to_chains
from em3na.utils.cryo_utils import parse_map, write_map

from em3na.na_utils.data import nucleotide_constants as nc

def add_args(parser):
    parser.add_argument("--pdb", "-p", help="Input structure")
    parser.add_argument("--seq", "-s", help="Input seq")
    parser.add_argument("--aamap", "-a", help="Input aa type map")
    return parser

def main(args):
    return assign(
        args.pdb,
        args.seq,
        args.aamap,
    )

def assign(
    dir_pdb,
    dir_seq,
    dir_aamap,
):
    # read input fasta file
    print("# Read seqs")
    seqs = read_fasta(dir_seq)
    seqs = [seq.upper() for seq in seqs]
    print("# Found {} seqs".format(len(seqs)))
    for k, seq in enumerate(seqs):
        print(">Seq_{}".format(k))
        print(seq)
    assert len(seqs) > 0, "# Num of seqs should be > 0"

    # read input structure
    print("# Read pdb")
    atom_pos, atom_mask, _, _, chain_idx, _ = read_pdb(dir_pdb)
    chains_atom_pos, chains_atom_mask = split_to_chains(chain_idx, atom_pos, atom_mask)
    print("# Read {} chains {} residues form {}".format(chain_idx.max() + 1, len(atom_pos), dir_pdb))

    # read aamap
    print("# Read aa map")
    aamap, origin, nxyz, voxel_size = parse_map(dir_aamap, False, None)

    # extract representative atoms - C4'
    chains_pos = [chain_atom_pos[..., 5] for chain_atom_pos in chains_atom_pos]

    return assign_chains(
        chains_pos,
        aamap,
        origin=origin,
        voxel_size=voxel_size,
    )


def assign_chains(
    chains_pos,
    seqs, 
    aamap, 
    origin=None,
    voxel_size=None,
):
    if origin is None:
        origin = np.zeros(3, dtype=np.float32)

    if voxel_size is None:
        voxel_size = np.ones(3, dtype=np.float32)

    # align
    idx_to_restypes = ["A", "G", "C", "U"]

    assigned_seqs = []
    for k, chain_pos in enumerate(chains_pos):
        # get sequence from structure using C4'
        seq1 = [
            get_aa_type(aamap, chain_pos[i], origin=origin) for i in range(len(chain_pos))
        ]
        seq1 = [
            idx_to_restypes[x] for x in seq1
        ]
        seq1 = "".join(seq1)

        # get sequence alignment
        seqA, seqB, score, k = get_seq_align_among_candidates(seq1, seqs)
        print("# Best fit seq is {}".format(seqB))

        # convet seqA to seq
        assigned_seqs.append(seqB)

    assigned_res_type = []
    for assigned_seq in assigned_seqs:
        assigned_res_type.append(
            1 + 20 + np.array([nc.restypes_1_to_index[x] for x in assigned_seq], dtype=np.int32), 
        )
    return assigned_res_type

    # concat all chains
    #assigned_seq = reduce(lambda x, y: x + y, assigned_seqs)
    #assigned_res_type = [nc.restypes_1_to_index[x] for x in assigned_seq]
    #assigned_res_type = np.array(assigned_res_type, dtype=np.int32)
    #print(assigned_res_type)
    #return assigned_res_type

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = add_args(parser).parse_args()
    main(args)

