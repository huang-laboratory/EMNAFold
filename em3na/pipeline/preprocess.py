# do quick preprocess

import os
import sys
import shutil
import numpy as np

from em3na.utils.misc_utils import pjoin, abspath
from em3na.io.seqio import read_fasta, filter_non_acgut, write_seqs_to_file

def main(args):
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    # Check input seq
    has_rna = False
    has_dna = False

    if args.rna is not None:
        # Read fasta
        seqs = read_fasta(args.rna)
        # Filter non-A, C, G, U, T types
        seqs = [filter_non_acgut(seq.upper()) for seq in seqs]
        # Filter empty seqs
        seqs = [seq for seq in seqs if len(seq) >= 1]
        if len(seqs) >= 1:
            # Re-write
            fo = pjoin(args.output, "format_seq_rna.fasta")
            write_seqs_to_file(seqs, fo)
            print("# Write formated RNA seq to {}".format(fo))
            has_rna = True
        else:
            print("# Although you input RNA seq but no valid seq is kept")
    else:
        print("# You do not input RNA seq")


    if args.dna is not None:
        # Read fasta
        seqs = read_fasta(args.dna)
        # Filter non-A, C, G, U, T types
        seqs = [filter_non_acgut(seq.upper()) for seq in seqs]
        # Filter empty seqs
        seqs = [seq for seq in seqs if len(seq) >= 1]
        if len(seqs) >= 1:
            # Re-write
            fo = pjoin(args.output, "format_seq_dna.fasta")
            write_seqs_to_file(seqs, fo)
            print("# Write formated DNA seq to {}".format(fo))
            has_dna = True
        else:
            print("# Although you input DNA seq but no valid seq is kept")
    else:
        print("# You do not input DNA seq")


    if (not has_rna) and (not has_dna):
        print("# Please input valid DNA/RNA sequence(s)")


    # interp map
    if args.map is not None:
        from em3na.utils.cryo_utils import parse_map, write_map
        from em3na.utils.torch_utils import get_device_names
        device = get_device_names(args.device)[0]
        data, origin, _, vsize = parse_map(args.map, False, 1.0, device=device)
        map_out = pjoin(args.output, "format_map.mrc")
        write_map(
            map_out, 
            data, 
            origin=origin, 
            voxel_size=[1., 1., 1.],
        )
        print("# Write formated map to {}".format(map_out))



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna", "-r", help="Input rna sequence")
    parser.add_argument("--dna", "-d", help="Input dna sequence")
    parser.add_argument("--map", "-m", help="Input map")
    parser.add_argument("--device", default="cpu", help="Interpolation using CPU or GPU, using GPU is much faster")
    parser.add_argument("--output", "-o", help="Output directory", default="./")
    args = parser.parse_args()
    main(args)
