import argparse
import numpy as np
from scipy.spatial import KDTree
from em3na.io.pdbio import read_pdb
import warnings
warnings.filterwarnings("ignore")

def add_args(parser):
    parser.add_argument("--target", "-t", help="Target structure")
    parser.add_argument("--query", "-q", help="Query structure")
    parser.add_argument("--eval-chain", action='store_true', help="Eval in chain level")
    parser.add_argument("-d", type=float, default=3.0, help="Distance cutoff")
    return parser

def eval_local(
    target_atom_pos, target_atom_mask, target_res_type, 
    query_atom_pos, query_atom_mask, query_res_type, 
    r=3.0, 
):
    # get correspondence
    target_c4_pos = target_atom_pos[..., 1, :]
    query_c4_pos = query_atom_pos[..., 1, :]
    tree = KDTree(target_c4_pos)

    idxs = tree.query_ball_point(query_c4_pos, r=r+1e-3)

    sorted_idxs = []
    sorted_distances = []
    for q_point, target_idxs in zip(query_c4_pos, idxs):
        target_idxs = np.asarray(target_idxs, dtype=np.int32)

        if len(target_idxs) == 0:
            sorted_idxs.append( [] )
            sorted_distances.append( [] )
            continue

        dists = np.linalg.norm(target_c4_pos[target_idxs] - q_point, axis=1)
        order = np.argsort(dists)
        sorted_idxs.append(np.asarray(target_idxs)[order])
        sorted_distances.append( dists[order] )

    correspondence = []
    target_idx_used = set()
    for query_idx, (target_idxs, dists) in enumerate(zip(sorted_idxs, sorted_distances)):
        unused_idx = None
        for target_idx in target_idxs:
            if target_idx in target_idx_used:
                continue
            else:
                unused_idx = target_idx
                break
        if unused_idx is not None:
            target_idx_used.add(unused_idx)
            correspondence.append( [query_idx, unused_idx] )

    correspondence = np.asarray(correspondence, dtype=np.int32)

    if len(correspondence) > 0:
        #p_rmsd = np.sqrt(
        #    np.mean(
        #        np.sum(
        #            np.power(
        #                target_atom_pos[correspondence[:, 1]][..., 8, :] - query_atom_pos[correspondence[:, 0]][..., 8, :], 
        #                2
        #            ), 
        #            axis=-1, 
        #        ),
        #    ),
        #)

        c4_rmsd = np.sqrt(
            np.mean(
                np.sum(
                    np.power(
                        target_c4_pos[correspondence[:, 1]] - query_c4_pos[correspondence[:, 0]], 
                        2
                    ), 
                    axis=-1, 
                ),
            ),
        )

        target_bb_pos = target_atom_pos[correspondence[:, 1]]
        target_bb_mask = target_atom_mask[correspondence[:, 1]]
        target_bb_mask[..., 11:] = False

        query_bb_pos = query_atom_pos[correspondence[:, 0]]
        query_bb_mask = query_atom_mask[correspondence[:, 0]]
        query_bb_mask[..., 11:] = False

        common_bb_mask = np.logical_and(target_bb_mask, query_bb_mask)

        bb_rmsd = np.sqrt(
            np.mean(
                np.sum(
                    np.power( target_bb_pos[common_bb_mask] - query_bb_pos[common_bb_mask], 2 ),
                    axis=-1, 
                )
            ),
        )

        cov = len(correspondence[:, 0]) / len(target_c4_pos)

        seq_match = np.sum(target_res_type[correspondence[:, 1]] == query_res_type[correspondence[:, 0]]) / len(correspondence[:, 0])

        seq_recall = cov * seq_match

        print("# len = {} C4_RMSD = {:.4f} BB_RMSD = {:.4f} Cov = {:.4f} Seq_Match = {:.4f} Seq_Recall = {:.4f}".format(len(query_atom_pos), c4_rmsd, bb_rmsd, cov, seq_match, seq_recall))
    else:
        print("# len = {} C4_RMSD = - BB_RMSD = - Cov = 0.0 Seq_Match = - Seq_Recall = - ".format( len(query_atom_pos) ))


def main(args):
    tgt_atom_pos, tgt_atom_mask, tgt_res_type, tgt_res_idx, tgt_chain_idx, _ = read_pdb(args.target)
    tgt_has_c4_atom = tgt_atom_mask[..., 1].astype(bool)
    tgt_atom_pos = tgt_atom_pos[tgt_has_c4_atom]
    tgt_atom_mask = tgt_atom_mask[tgt_has_c4_atom]
    tgt_res_type = tgt_res_type[tgt_has_c4_atom]
    tgt_res_idx = tgt_res_idx[tgt_has_c4_atom]
    tgt_chain_idx = tgt_chain_idx[tgt_has_c4_atom]

    qry_atom_pos, qry_atom_mask, qry_res_type, qry_res_idx, qry_chain_idx, _ = read_pdb(args.query)
    qry_has_c4_atom = qry_atom_mask[..., 1].astype(bool)
    qry_atom_pos = qry_atom_pos[qry_has_c4_atom]
    qry_atom_mask = qry_atom_mask[qry_has_c4_atom]
    qry_res_type = qry_res_type[qry_has_c4_atom]
    qry_res_idx = qry_res_idx[qry_has_c4_atom]
    qry_chain_idx = qry_chain_idx[qry_has_c4_atom]

    print("# Read {} valid residues from {}".format(len(tgt_atom_pos), args.target))
    print("# Read {} valid residues from {}".format(len(qry_atom_pos), args.query))

    # convert DA DC DG DT to A C G U
    tgt_res_type[tgt_res_type == 21] = 25
    tgt_res_type[tgt_res_type == 22] = 26
    tgt_res_type[tgt_res_type == 23] = 27
    tgt_res_type[tgt_res_type == 24] = 28

    qry_res_type[qry_res_type == 21] = 25
    qry_res_type[qry_res_type == 22] = 26
    qry_res_type[qry_res_type == 23] = 27
    qry_res_type[qry_res_type == 24] = 28

    print("# Using distance cutoff = {:.4f}".format(args.d))

    # individual chains
    if args.eval_chain:
        qry_n_chain = qry_chain_idx.max() + 1
        for i in range(qry_n_chain):
            sel_idxs = np.arange(0, len(qry_atom_pos))[qry_chain_idx == i]
            print("# Eval chain {} len = {}".format(i, len(sel_idxs)))

            eval_local(
                tgt_atom_pos,
                tgt_atom_mask,
                tgt_res_type,
                qry_atom_pos[sel_idxs],
                qry_atom_mask[sel_idxs],
                qry_res_type[sel_idxs], 
                r=args.d, 
            )


    # all residues
    print("# Eval all len = {}".format(len(qry_atom_pos)))
    eval_local(
        tgt_atom_pos,
        tgt_atom_mask,
        tgt_res_type,
        qry_atom_pos,
        qry_atom_mask,
        qry_res_type,
        r=args.d, 
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = add_args(parser).parse_args()
    main(args)
