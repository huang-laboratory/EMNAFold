"""Main program"""
import os
import sys
import time
import shutil
import argparse
import numpy as np

from em3na.io.pdbio import fix_quotes
from em3na.utils.misc_utils import pjoin, abspath

def add_args(parser):
    parser.add_argument("--map", "-m", help="Input map", required=True)
    parser.add_argument("--rna", "-r", help="Input rna sequence")
    parser.add_argument("--dna", "-d", help="Input dna sequence")
    # Using --dna/--rna instead of a consensus --seq
    #parser.add_argument("--seq", "-s", help="Input sequence")
    parser.add_argument("--output", "-o", help="Output directory", required=True)
    parser.add_argument("--device", "--gpu", help="GPU device", default="0")
    parser.add_argument("--keep-temp-files", action="store_true")
    # Resolution (not used by now)
    #parser.add_argument("--resolution", "-r", help="Map resolution", type=float)

    # Skipping controls
    skip_group = parser.add_argument_group("Skipping options")
    skip_group.add_argument("--skip_preprocess", action='store_true', help=argparse.SUPPRESS)
    skip_group.add_argument("--skip_c4",         action='store_true', help=argparse.SUPPRESS)
    skip_group.add_argument("--skip_aa",         action='store_true', help=argparse.SUPPRESS)
    skip_group.add_argument("--skip_denovo",     action='store_true', help=argparse.SUPPRESS)
    skip_group.add_argument("--skip_map_to_p",   action='store_true', help=argparse.SUPPRESS)
    skip_group.add_argument("--skip_infer",      action='store_true', help=argparse.SUPPRESS)
    skip_group.add_argument("--skip_infer_aa",   action='store_true', help=argparse.SUPPRESS)

    return parser

def main(args):
    print("# Start modeling")

    script_dir = os.path.dirname(__file__)
    model_dir = pjoin(script_dir, "weights")

    out_dir = abspath(args.output)
    temp_dir = pjoin(out_dir, "temp")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    # preprocess
    if not args.skip_preprocess:
        start = time.time()
        from em3na.pipeline import preprocess
        preprocess_args = argparse.Namespace()
        preprocess_args.map = args.map
        preprocess_args.rna = args.rna
        preprocess_args.dna = args.dna
        preprocess_args.output = temp_dir

        preprocess_args.device = args.device

        preprocess.main(preprocess_args)
        end = time.time()
        print("# Time = {:.4f}".format(end - start))
    else:
        print("# Skip preprocess")

    # predict c4 atoms
    if not args.skip_c4:
        start = time.time()
        from em3na.pipeline import pred
        pred_args = argparse.Namespace()
        pred_args.input = pjoin(temp_dir, "format_map.mrc")
        pred_args.output = pjoin(temp_dir, "pred")
        pred_args.contour = 1e-6
        pred_args.batchsize = 40
        pred_args.device = args.device
        pred_args.model = os.path.join(model_dir)
        pred_args.stride = 12

        pred.main(pred_args)
        end = time.time()
        print("# Time = {:.4f}".format(end - start))
    else:
        print("# Skip dl c4")

    # predict a vanilla aa map
    if not args.skip_aa:
        start = time.time()
        from em3na.pipeline import pred_aa
        pred_aa_args = argparse.Namespace()
        pred_aa_args.input = pjoin(temp_dir, "pred", "na.mrc")
        pred_aa_args.output = pjoin(temp_dir, "pred")
        pred_aa_args.contour = 1e-6
        pred_aa_args.batchsize = 40
        pred_aa_args.device = args.device
        pred_aa_args.model = os.path.join(model_dir)
        pred_aa_args.stride = 16

        pred_aa.main(pred_aa_args)
        end = time.time()
        print("# Time = {:.4f}".format(end - start))
    else:
        print("# Skip dl aa")

    # ms
    if not args.skip_denovo:
        if not args.skip_map_to_p:
            start = time.time()
            from em3na.pipeline import pmap_to_p
            pmap_to_p_args = argparse.Namespace()
            pmap_to_p_args.map = pjoin(temp_dir, "pred", "c4.mrc")
            pmap_to_p_args.p = None
            pmap_to_p_args.output = pjoin(temp_dir, "pred")
            pmap_to_p_args.lib = script_dir
            
            pmap_to_p_args.thresh = 3.0
            pmap_to_p_args.res = 6.0
            pmap_to_p_args.nt = 4
            pmap_to_p_args.filter = 0.0
            pmap_to_p_args.dmerge = 3.5

            # process on gpu will be much faster
            pmap_to_p_args.device = args.device

            pmap_to_p.main(pmap_to_p_args)
            end = time.time()
            print("# Time = {:.4f}".format(end - start))
        else:
            print("# Skip map to p")

        # gnn
        if not args.skip_infer:
            start = time.time()
            from em3na.infer import inference
            inference_args = argparse.Namespace()
            inference_args.map = pjoin(temp_dir, "format_map.mrc")
            inference_args.struct = pjoin(temp_dir, "pred", "raw_c4.pdb")

            inference_args.model_dir = pjoin(model_dir, "all_atom")

            inference_args.output_dir = pjoin(temp_dir, "denovo")
            inference_args.device = args.device
            inference_args.crop_length = 128
            inference_args.repeat_per_residue = 1
            inference_args.refine = False
            inference_args.batch_size = 1
            inference_args.fp16 = False
            inference_args.voxel_size = 1.0

            inference_args.recycle = 3
            inference_args.no_use_random_affine = False

            inference.main(inference_args)
            end = time.time()
            print("# Time = {:.4f}".format(end - start))
        else:
            print("# Skip infer all atom")

        # gnn aa
        if not args.skip_infer_aa:
            # run RNA seq embedding
            print("# Run seq embedding")
            start = time.time()
            from em3na.pipeline import run_seq_embed
            run_seq_embed_args = argparse.Namespace()
            run_seq_embed_args.rna = pjoin(temp_dir, "format_seq_rna.fasta")
            run_seq_embed_args.dna = pjoin(temp_dir, "format_seq_dna.fasta")

            run_seq_embed_args.max_length = 1000
            run_seq_embed_args.device = args.device
            run_seq_embed_args.model_dir = pjoin(script_dir, "rinalmo", "weights")
            run_seq_embed_args.model_name = "rinalmo_giga_pretrained.pt"
            run_seq_embed_args.output_dir = pjoin(temp_dir, "denovo")
            run_seq_embed.main(run_seq_embed_args)
            end = time.time()
            print("# Time = {:.4f}".format(end - start))

            # run pseudo SS parsing
            print("# Parsing sec strs")
            from em3na.pipeline import parse_sec_str
            parse_sec_str_args = argparse.Namespace()
            parse_sec_str_args.struct = os.path.join(temp_dir, "denovo", "backbone.cif")
            parse_sec_str_args.output_dir = pjoin(temp_dir, "denovo", "secstr")
            parse_sec_str_args.temp_dir = pjoin(temp_dir, "denovo", "secstr")
            parse_sec_str.main(parse_sec_str_args)

            # run gnn aa
            print("# Inference aa")
            start = time.time()
            from em3na.infer import inference_aa
            inference_aa_args = argparse.Namespace()
            inference_aa_args.map = pjoin(temp_dir, "format_map.mrc")
            inference_aa_args.struct = pjoin(temp_dir, "denovo", "backbone.cif")
            inference_aa_args.sec_str = os.path.join(temp_dir, "denovo", "secstr", "pairing_matrix_new.npy")
            inference_aa_args.seq_embed = pjoin(temp_dir, "denovo", "seq_embed.npy")
            inference_aa_args.model_dir = pjoin(model_dir, "all_atom_aa")
            inference_aa_args.output_dir = pjoin(temp_dir, "denovo")
            inference_aa_args.device = args.device

            inference_aa_args.voxel_size = 1.0
            inference_aa_args.rna = pjoin(temp_dir, "format_seq_rna.fasta")
            inference_aa_args.dna = pjoin(temp_dir, "format_seq_dna.fasta")

            inference_aa_args.affines = pjoin(temp_dir, "denovo", "affines.npy")
            inference_aa_args.torsions = pjoin(temp_dir, "denovo", "torsions.npy")

            # vanilla logits
            inference_aa_args.npz = pjoin(temp_dir, "pred", "logits.npz")

            inference_aa.main(inference_aa_args)

            end = time.time()
            print("# Time = {:.4f}".format(end - start))
        else:
            print("# Skip infer all atom aa")

    else:
        print("# Skip denovo modeling")


    # Copy final models
    has_output = False
    src = pjoin(temp_dir, "denovo", "denovo.cif")
    dst = pjoin(out_dir, "output.cif")
    if os.path.exists(src):
        shutil.copy(src, dst)
        fix_quotes(dst)
        has_output = True

    # Remove temp files
    if not args.keep_temp_files:
        print("# No keep temp files")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        else:
            pass
    else:
        print("# Keep temp files")

    if has_output:
        print("# Check modeling result at {}".format(dst))
    else:
        print("# Modeling fails")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = add_args(parser).parse_args()
    main(args)
