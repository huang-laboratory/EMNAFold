import argparse
import numpy as np

from em3na.utils.qscore.q_score import calculate_q_score
from em3na.utils.qscore.mrc_utils import load_mrc
from em3na.utils.qscore.pdb_utils import get_protein_from_file_path

def add_args(parser):
    parser.add_argument("--struct", "-s")
    parser.add_argument("--map", "-m")
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = add_args(parser).parse_args()
    main(args)

def main(args):
    np.random.seed(42)

    # read structure
    prot = get_protein_from_file_path(args.struct)

    # read map
    map = load_mrc(args.map)

    mask = prot.atom_mask.astype(bool)

    atoms = prot.atom_positions[mask]
    q_scores = calculate_q_score(atoms, map)
    q_score_per_residue = np.zeros_like(mask, dtype=np.float32)
    q_score_per_residue[mask] = q_scores
    q_score_per_residue = q_score_per_residue.sum(axis=1) / (1e-6 + mask.sum(axis=1))
    avg_q_score = np.mean(q_scores)
    min_q_score = np.min(q_scores)
    max_q_score = np.max(q_scores)
    print(f"Mean {avg_q_score:.6f} Min {min_q_score:.6f} Max {max_q_score:.6f}")

