import os
import sys
import time
import argparse
import subprocess
import numpy as np

from em3na.io.seqio import read_lines
from em3na.io.pdbio import read_pdb, chains_atom_pos_to_pdb, chains_atom_pos_to_data
from em3na.utils.misc_utils import get_temp_dir
from em3na.ss_utils.ss import dbn_to_pairing_matrix, get_pair_indices_numpy

def backbone_to_sec_str_x(atom_pos, atom_mask, chain_idx, lib_dir='.', temp_dir='.', na='rna'):
    if na.lower() == "dna":
        offset = 21
    elif na.lower() == "rna":
        offset = 21 + 4
    else:
        raise NotImplementedError

    dummy_atom_mask = atom_mask.copy()
    dummy_atom_mask[..., 11:] = False

    dummy_atom_pos = atom_pos.copy() + 1e-2

    res_type = np.ones( (len(atom_pos),), dtype=np.int32 ) * 0 + offset
    dummy_res_type = np.ones( (len(dummy_atom_pos),), dtype=np.int32 ) * 3 + offset

    with get_temp_dir(temp_dir) as __temp_dir:
        # Write common backbone to file
        fpdb = os.path.join(__temp_dir, f"concated.data")
        chains_atom_pos_to_data(
            fpdb,
            [atom_pos, dummy_atom_pos],
            [dummy_atom_mask, dummy_atom_mask],
            [res_type, dummy_res_type],
        )

        # PDB files do not support > 100000 atoms
        # So we write individual chains and than merge
        """
        res_idx = np.arange(0, len(atom_pos))
        filenames = []
        n_chain = chain_idx.max() + 1
        for i in range(n_chain):
            fpdb = os.path.join(__temp_dir, f"all_A_chain_{i}.pdb")
            filenames.append(fpdb)

            chains_atom_pos_to_pdb(
                fpdb,
                [atom_pos[chain_idx == i]],
                [dummy_atom_mask[chain_idx == i]],
                [res_type[chain_idx == i]],
                [res_idx[chain_idx == i]],
                #[2 * i],
                suffix='pdb',
            )
        for i in range(n_chain):
            fpdb = os.path.join(__temp_dir, f"all_U_chain_{i}.pdb")
            filenames.append(fpdb)

            chains_atom_pos_to_pdb(
                fpdb,
                [dummy_atom_pos[chain_idx == i]],
                [dummy_atom_mask[chain_idx == i]],
                [dummy_res_type[chain_idx == i]],
                [res_idx[chain_idx == i]],
                #[2 * i + 1], 
                suffix='pdb',
            )
        # Concat
        all_lines = []
        for filename in filenames:
            lines = read_lines(filename)
            all_lines.extend(lines)
        
        fpdb = os.path.join(__temp_dir, f"concated.pdb")
        with open(fpdb, 'w') as f:
            for line in all_lines:
                f.write( line.strip("\n") + "\n" )
        """

        # Run CSSRX
        fcssr = os.path.join(__temp_dir, f"concated.cssr")
        cmd = [os.path.join(lib_dir, "bin", "CSSRX"), fpdb, fcssr, "-o", "1"]
        #print(cmd)

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        #print(result)

        with open(fcssr, 'r') as f:
            dbn = f.readline().strip()
        #print(dbn)

        # Get a sub matrix
        pairing_matrix = dbn_to_pairing_matrix(dbn).astype(np.int32)

    return pairing_matrix

def add_args(parser):
    parser.add_argument("--struct", type=str, required=True)
    parser.add_argument("--output-dir", "-o", type=str)
    parser.add_argument("--temp-dir")
    return parser

def main(args):
    atom_pos, atom_mask, res_type, res_idx, chain_idx, bfactors = read_pdb(args.struct)
    print("# Read {} residues {} chains from {}".format(len(atom_pos), chain_idx.max() + 1, args.struct))

    script_dir = os.path.abspath(os.path.dirname(__file__))
    lib_dir = os.path.join(script_dir, "..")

    temp_dir = args.temp_dir
    os.makedirs(temp_dir, exist_ok=True)

    pairing_matrix = backbone_to_sec_str_x(
        atom_pos, 
        atom_mask, 
        chain_idx, 
        temp_dir=temp_dir,
        lib_dir=lib_dir, 
        na="rna",
    )
    pairs = get_pair_indices_numpy(pairing_matrix)
    new_pairs = set()
    for p in pairs:
        i, k = p
        if i >= len(atom_pos):
            i -= len(atom_pos)
        if k >= len(atom_pos):
            k -= len(atom_pos)

        if i >= k:
            i, k = k, i

        new_pairs.add((i, k))

    new_pairs = list(new_pairs)
    new_pairs.sort(key=lambda x:x[0])


    # save new pairing matrix
    l = len(atom_pos)
    new_pairing_matrix = np.zeros(
        (l, l),
        dtype=bool, 
    )

    for p in new_pairs:
        i, k = p
        new_pairing_matrix[i, k] = True
        new_pairing_matrix[k, i] = True

        ci, ck = chain_idx[i], chain_idx[k]
        ri, rk = res_idx[i], res_idx[k]

        #print("{} {} - {} {}".format(ci, ri, ck, rk))


    # save
    fo = os.path.join(args.output_dir, "pairing_matrix.npy")
    np.save(fo, pairing_matrix.astype(bool))
    print("# Save pairing frequency matrix npy (with shape {}) to {}".format(pairing_matrix.shape, fo))

    # save
    fo = os.path.join(args.output_dir, "pairing_matrix_new.npy")
    np.save(fo, new_pairing_matrix.astype(bool))
    print("# Save new pairing frequency matrix npy (with shape {}) to {}".format(new_pairing_matrix.shape, fo))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = add_args(parser).parse_args()
    main(args)
