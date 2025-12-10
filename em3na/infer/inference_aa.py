import os
import sys
import tqdm
import time
import torch
import warnings
warnings.filterwarnings("ignore")
import argparse
import numpy as np
from collections import namedtuple
import pickle
from em3na.utils.cryo_utils import parse_map, write_map, enlarge_grid, MRCObject
from em3na.infer.torch_utils import get_device_names
from em3na.infer.to_all_atom import affines_and_torsions_to_all_atom
from em3na.modules.model_aa_recycle import Model
from em3na.io.pdbio import read_pdb, chains_atom_pos_to_pdb, split_to_chains, atom27_to_atom23

from em3na.utils.affine_utils import affine_from_3_points
from em3na.io.seqio import read_fasta, is_na, is_all_same
from em3na.utils.misc_utils import pjoin, abspath, seed_everything
from em3na.utils.hmm.hmm_sequence_align import fix_chains_pipeline, FixChainsOutput
from em3na.utils.hmm.match_to_sequence import MatchToSequence

from em3na.infer.fixing import fix_match

from em3na.infer.to_all_atom import (
    affines_and_torsions_to_all_atom,
)

from em3na.utils import residue_constants as rc

def to_tensor(x, device='cpu'):
    return torch.from_numpy(x).to(device)

def safe_softmax(x, dim):
    x_max, _ = torch.max(x, dim=dim, keepdim=True)
    x_exp = torch.exp(x - x_max)
    x_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_sum

# Infer aatype for each residue
# Align with input sequences

def add_args(parser):
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", "--i", required=True, help="The path to the input map")
    parser.add_argument(
        "--struct", "--s", required=True, help="The path to the structure file"
    )
    parser.add_argument(
        "--sec-str", "--sec", required=True, 
        help="Secondary structure of target structure",
    )
    parser.add_argument(
        "--seq-embed",
        help="Embedding of target sequence",
    )
    parser.add_argument("--model-dir", required=True, help="Where the model at")
    parser.add_argument("--output-dir", "-o", default=".", help="Where to save the results")
    parser.add_argument("--device", default="cpu", help="Which device to run on")
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=1.0,
        help="The voxel size that the GNN should be interpolating to."
    )
    # for sequence
    parser.add_argument(
        "--rna",
        type=str,
        help="Input dna sequence"
    )
    parser.add_argument(
        "--deq",
        type=str,
        help="Input rna sequence"
    )
    #parser.add_argument(
    #    "--seq",
    #    type=str,
    #    help="Input sequence"
    #)
    # for all-atom construction
    parser.add_argument(
        "--affines",
        type=str
    )
    # for aa logits npz
    parser.add_argument(
        "--npz", "-n", 
        type=str,
    )
    parser.add_argument(
        "--torsions",
        type=str, 
    )
    return parser


def main(args):
    seed_everything(42)

    # Set output dir
    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = os.path.dirname(args.output_dir)

    device_names = get_device_names(args.device)
    num_devices = len(device_names)

    # Read structure
    if args.struct.endswith(".cif") or args.struct.endswith(".pdb"):
        atom_pos, atom_mask, res_type, res_idx, chain_idx, bfactor = struct = read_pdb(args.struct)
    else:
        raise RuntimeError(f"File {args.struct} is not a supported file format.")
    print("# Done load structure with {} residues".format(len(atom_pos)))

    # Read sec str : pairing matrix
    pairing_matrix = None
    if args.sec_str.endswith(".pkl"):
        with open(args.sec_str, 'rb') as f:
            pairing_matrix = pickle.load(f)['pairing_matrix']
    elif args.sec_str.endswith(".npy"):
        pairing_matrix = np.load(args.sec_str)
    elif args.sec_str.endswith(".dbn"):
        raise NotImplementedError
    else:
        pass
    if pairing_matrix is None:
        raise RuntimeError("Please input a pairing matrix file of target structure")
    print("# Done load sec str with shape {}".format(pairing_matrix.shape))


    # Read sequence and run seq embedding
    rna_seqs = []
    if os.path.exists(args.rna):
        rna_seqs = read_fasta(args.rna)
        rna_seqs = [seq.upper() for seq in rna_seqs if is_na(seq)]

    dna_seqs = []
    if os.path.exists(args.dna):
        dna_seqs = read_fasta(args.dna)
        dna_seqs = [seq.upper() for seq in dna_seqs if is_na(seq)]

    # simple concat
    seqs = rna_seqs + dna_seqs
    seqs_is_dna = [False] * len(rna_seqs) + [True] * len(dna_seqs)
    assert len(seqs) > 0

    # convert all "T" to "U"
    seqs = [seq.replace("T", "U") for seq in seqs]

    # special cases
    first_character = seqs[0][0]
    seqs_is_all_same = np.all(
        np.asarray([is_all_same(seq, first_character) for seq in seqs], dtype=bool)
    )
    print("# Seqs is all same = {}".format(seqs_is_all_same))

    seq = "".join(seqs)
    new_seq = []
    for s in seq:
        if s == "T":
            s = "U"
        if s not in ["A", "C", "G", "U"]:
            s = "N"
        new_seq.append(s)
    seq = "".join(new_seq)
    print("# Done load seq")
    print("# Total seq length = {}".format(len(seq)))

    # Run seq embedding
    if args.seq_embed is not None:
        seq_embed = np.load(args.seq_embed)
        print("# Read precomputed seq embed")
    else:
        # TODO
        # run seq embed
        pass


    # Read map
    if not (args.map.endswith(".mrc") or args.map.endswith(".map")):
        warnings.warn(f"The file {args.map} does not end with '.mrc' or '.map'\nPlease make sure it is an MRC file.")
    grid, origin, _, voxel_size = parse_map(args.map, False, args.voxel_size)

    # Process grid (this does not change grid origin)
    maximum = np.percentile(grid[grid > 0.0], 99.999)
    grid = np.clip(grid, a_min=0.0, a_max=maximum)
    grid = grid / (maximum + 1e-6)
    grid = enlarge_grid(grid)
    print("# Done load map")


    # Get model
    model = Model(
        n_block=6, 
        use_checkpoint=False,
    )
    model_state_dict = torch.load(args.model_dir, map_location='cpu')
    model.load_state_dict(model_state_dict)
    model.eval()
    model.to(device_names[0]) # only use device 0
    print("# Done loading model")


    # Convert feats to tensor
    seq_embed = to_tensor(seq_embed, device_names[0]).float()
    grid = to_tensor(grid, device_names[0]).float()
    origin = to_tensor(origin, device_names[0]).float()
    voxel_size = to_tensor(voxel_size, device_names[0]).float()

    chain_index = to_tensor(chain_idx, device_names[0]).long()
    residue_index = to_tensor(res_idx, device_names[0]).long()

    chain_idx_np = chain_idx.copy()
    res_idx_np = res_idx.copy()

    atom_pos_np = atom_pos.copy()
    atom_pos = to_tensor(atom_pos, device_names[0]).float()

    affines = affine_from_3_points(
        atom_pos[..., 2, :],
        atom_pos[..., 1, :],
        atom_pos[..., 0, :],
    )
    affines[..., :, 0] = affines[..., :, 0] * (-1)
    affines[..., :, 2] = affines[..., :, 2] * (-1)

    # Run inference
    tstart = time.time()
    n = len(atom_pos)

    n_iter = 3
    length = 256
    stride = 64
    with torch.no_grad():
        # For recycling
        use_aatype_probs = torch.nn.functional.one_hot(
            torch.zeros( (len(affines), ) ).to(device_names[0]).long(),
            num_classes=2,
        ).float()
        pred_aatype_probs = torch.zeros(
           (len(affines), 4)
        ).to(device_names[0]).float()

        # averaged aa type logits (for too long sequence)
        pred_aatype_sum = torch.zeros( (len(atom_pos), 4) ).to(device_names[0]).float()
        pred_aatype_cnt = torch.zeros( (len(atom_pos),  ) ).to(device_names[0]).int()

        for i_iter in range(n_iter):
            print("# Iteration {} / {}".format(i_iter + 1, n_iter))
            for i in tqdm.trange(0, n, stride):
                start = i
                end = start + length
                if end > n:
                    end = n

                #print("# Infer from {} to {}".format(start, end))
                idxs = list(range(start, end))

                sub_pairing_matrix = pairing_matrix[np.ix_(idxs, idxs)]
                sub_pairing_matrix = to_tensor(sub_pairing_matrix, device_names[0]).float()

                # At least 2 points (1 edge)
                if len(idxs) >= 2:
                    # Infer
                    output = model(
                        aa_probs=torch.cat(
                            [pred_aatype_probs[idxs], use_aatype_probs[idxs]],
                            dim=-1,
                        ),
                        affines=affines[idxs], # selected ones
                        seq_embed=seq_embed[None, ...], # full-length seq embed
                        seq_idx=residue_index[idxs], # selected ones
                        chain_idx=chain_index[idxs], # selected ones
                        pairing_matrix=sub_pairing_matrix, # selected ones
                        # density
                        cryo_grids=[grid[None, None, ...]],
                        cryo_global_origins=[origin[None, ...]],
                        cryo_voxel_sizes=[voxel_size[None, ...]],
                        batch=None, 
                    )

                    pred_aatype_sum[idxs] += output['pred_aatype']
                    pred_aatype_cnt[idxs] += 1

            # Get mean logits
            pred_aatype_cnt[pred_aatype_cnt < 1] = 1
            pred_aatype = pred_aatype_sum / pred_aatype_cnt[..., None]

             # Softmax
            pred_aatype_probs = safe_softmax(pred_aatype, dim=-1)
            use_aatype_probs = torch.nn.functional.one_hot(
                torch.ones( (len(affines), ) ).to(device_names[0]).long(),
                num_classes=2,
            ).float()

    # To numpy
    pred_aatype = pred_aatype.cpu().numpy()

    # get mean logits
    pred_aatype_cnt[pred_aatype_cnt < 1] = 1
    pred_aatype = pred_aatype_sum / pred_aatype_cnt[..., None]
    pred_aatype = pred_aatype.cpu().numpy()

    # save logits
    save_dir = os.path.join(args.output_dir, "logits_tta.npy")
    np.save(save_dir, pred_aatype)
    print("# Save pred TTA logits to {}".format(save_dir))


    # aa logits for all type
    pred_aatype_all = np.ones( (n, 28), dtype=np.float32 ) * -100.0
    pred_aatype_all[..., 24:28] = pred_aatype # rna and dna is the same
    pred_aatype_all[..., 20:24] = pred_aatype # rna and dna is the same

    # hmm alignment
    hmm_temp_dir = os.path.join(args.output_dir, "hmm")
    os.makedirs(hmm_temp_dir, exist_ok=True)


    n_chain = chain_idx.max() + 1
    chains = []
    chains_aa_logits = []
    chains_prot_mask = []
    ca_pos = atom_pos_np[..., 1, :]
    all_idxs = np.arange( n, dtype=np.int32 )

    for i in range(n_chain):
        i_mask = chain_idx == i
        chains.append( all_idxs[i_mask] )
        chains_prot_mask.append( np.zeros( (i_mask.sum(), ), dtype=bool) )
        chains_aa_logits.append( pred_aatype_all[i_mask] )

    ############################
    ### HMM align for TTA logits
    ############################
    print("# HMM alignment for TTA logits")
    fix_chains_output = fix_chains_pipeline(
        prot_sequences=[],
        rna_sequences=seqs, # use RNA/DNA hybrid sequences, no split
        dna_sequences=[],
        chains=chains,
        chain_aa_logits=chains_aa_logits,
        ca_pos=ca_pos,
        chain_prot_mask=chains_prot_mask,
        chain_confidences=None,
        base_dir=hmm_temp_dir, 
        sort=False, 
    )


    ################################
    ### HMM align for Vanilla logits
    ################################
    if args.npz is not None:
        data = np.load(args.npz)
        grid = data['map']

        origin = data['origin']
        voxel_size = data['voxel_size']

        # Get logits
        logits_pred = []
        for i in range(len(atom_pos_np)):
            c4_pos = atom_pos_np[i][1, :]
            xyz = np.round((c4_pos - origin) / voxel_size).astype(np.int32)
            logits = grid[:, xyz[2], xyz[1], xyz[0]]
            logits_pred.append(logits)
        logits_pred = np.asarray(logits_pred, dtype=np.float32)

        # save logits
        save_dir = os.path.join(args.output_dir, "logits_vanilla.npy")
        np.save(save_dir, logits_pred)
        print("# Save pred vanilla logits to {}".format(save_dir))


        logits_all = np.ones( (len(logits_pred), 28), dtype=np.float32 ) * -100
        logits_all[:, 24:28] = logits_pred # rna and dna is the same
        logits_all[:, 20:24] = logits_pred # rna and dna is the same

        # Split logits to chains
        chains_aa_logits_vanilla = []
        for i in range(n_chain):
            i_mask = chain_idx == i
            chains_aa_logits_vanilla.append( logits_all[i_mask] )

        # HMM
        print("# HMM alignment for vanilla logits")
        fix_chains_output_vanilla = fix_chains_pipeline(
            prot_sequences=[],
            rna_sequences=seqs, # use RNA/DNA hybrid sequences, no split
            dna_sequences=[],
            chains=chains,
            chain_aa_logits=chains_aa_logits_vanilla,
            ca_pos=ca_pos,
            chain_prot_mask=chains_prot_mask,
            chain_confidences=None,
            base_dir=hmm_temp_dir,
            sort=False, 
        )

        #print(fix_chains_output.best_match_output.match_scores)
        s = np.sum(fix_chains_output.best_match_output.match_scores)
        print("# TTA score = {:.4f}".format(s))
        #print(fix_chains_output.chains[0][0])

        #print(fix_chains_output_vanilla.best_match_output.match_scores)
        s_vanilla = np.sum(fix_chains_output_vanilla.best_match_output.match_scores)
        print("# Vanilla score = {:.4f}".format(s_vanilla))
        #print(fix_chains_output_vanilla.chains[0][0])

        # Choose best among chains
        # Two gives the same chain idxs
        chains = fix_chains_output.chains

        x_new_sequences = []
        x_residue_idxs = []
        x_sequence_idxs = []
        x_key_start_matches = []
        x_key_end_matches = []
        x_match_scores = []
        x_hmm_output_match_sequences = []
        x_exists_in_sequence_mask = []
        x_is_nucleotide = []

        s_cnt = 0
        s_vanilla_cnt = 0
        s_all = 0
        print("# Comparing logits")
        for i in range(len(chains)):
            s = fix_chains_output.best_match_output.match_scores[i]
            s_vanilla = fix_chains_output_vanilla.best_match_output.match_scores[i]
            s_all += len(chains[i])

            #print(s, s_vanilla)
            if s > s_vanilla:
                s_cnt += len(chains[i])

                #print("# Using TTA len = {}".format(len(chains[i])))
                x_new_sequences.append(
                    fix_chains_output.best_match_output.new_sequences[i]
                )
                x_residue_idxs.append(
                    fix_chains_output.best_match_output.residue_idxs[i]
                )
                x_sequence_idxs.append(
                    fix_chains_output.best_match_output.sequence_idxs[i]
                )
                x_key_start_matches.append(
                    fix_chains_output.best_match_output.key_start_matches[i]
                )
                x_key_end_matches.append(
                    fix_chains_output.best_match_output.key_end_matches[i]
                )
                x_match_scores.append(
                    fix_chains_output.best_match_output.match_scores[i]
                )
                x_hmm_output_match_sequences.append(
                    fix_chains_output.best_match_output.hmm_output_match_sequences[i]
                )
                x_exists_in_sequence_mask.append(
                    fix_chains_output.best_match_output.exists_in_sequence_mask[i]
                )
                x_is_nucleotide.append(
                    fix_chains_output.best_match_output.is_nucleotide[i]
                )
            else:
                s_vanilla_cnt += len(chains[i])

                #print("# Using vanilla len = {}".format(len(chains[i])))
                x_new_sequences.append(
                    fix_chains_output_vanilla.best_match_output.new_sequences[i]
                )
                x_residue_idxs.append(
                    fix_chains_output_vanilla.best_match_output.residue_idxs[i]
                )
                x_sequence_idxs.append(
                    fix_chains_output_vanilla.best_match_output.sequence_idxs[i]
                )
                x_key_start_matches.append(
                    fix_chains_output_vanilla.best_match_output.key_start_matches[i]
                )
                x_key_end_matches.append(
                    fix_chains_output_vanilla.best_match_output.key_end_matches[i]
                )
                x_match_scores.append(
                    fix_chains_output_vanilla.best_match_output.match_scores[i]
                )
                x_hmm_output_match_sequences.append(
                    fix_chains_output_vanilla.best_match_output.hmm_output_match_sequences[i]
                )
                x_exists_in_sequence_mask.append(
                    fix_chains_output_vanilla.best_match_output.exists_in_sequence_mask[i]
                )
                x_is_nucleotide.append(
                    fix_chains_output_vanilla.best_match_output.is_nucleotide[i]
                )

        print("# Using TTA ratio = {:.4f} using vanilla ratio = {:.4f}".format(
            s_cnt / s_all,
            s_vanilla_cnt / s_all, 
        ))

        # Gather 
        new_fix_chains_output = FixChainsOutput(
            chains=chains, 
            best_match_output=MatchToSequence(
                new_sequences=x_new_sequences,
                residue_idxs=x_residue_idxs,
                sequence_idxs=x_sequence_idxs,
                key_start_matches=np.array(x_key_start_matches),
                key_end_matches=np.array(x_key_end_matches),
                match_scores=np.array(x_match_scores),
                hmm_output_match_sequences=x_hmm_output_match_sequences,
                exists_in_sequence_mask=x_exists_in_sequence_mask,
                is_nucleotide=x_is_nucleotide,

            ), 
            unmodelled_sequences=None,
        )
        fix_chains_output = new_fix_chains_output

    print("# Fixing match")
    new_match = fix_match(
        fix_chains_output.best_match_output, 
        [seq.lower() for seq in seqs],
        match_score_cutoff=0.40,
        len_cutoff=6, 
    )

    # Get final result
    chains_res_type = []
    for k in range(len( new_match.new_sequences )):
        aatype = new_match.new_sequences[k]
        seq_idx = new_match.sequence_idxs[k]

        aatype[aatype == 29] = 1 # dna cases
        aatype[aatype == 20] = 0 # dna cases
        aatype[aatype == 21] = 1 # dna cases
        aatype[aatype == 22] = 2 # dna cases
        aatype[aatype == 23] = 3 # dna cases

        aatype[aatype == 30] = 1 # rna cases
        aatype[aatype == 24] = 0 # rna cases
        aatype[aatype == 25] = 1 # rna cases
        aatype[aatype == 26] = 2 # rna cases
        aatype[aatype == 27] = 3 # rna cases

        if seqs_is_dna[seq_idx]:
            aatype += 21 # dna cases
        else:
            aatype += 21 + 4 # rna cases

        chains_res_type.append(aatype)

    chains_atom_pos = []
    chains_atom_mask = []
    chains_res_idx = []
    chains_bfactor = []
    atom_mask[..., 11:] = False

    bfactor_np = np.repeat(bfactor[..., 1][..., None], 23, axis=-1)
    for chain in fix_chains_output.chains:
        chains_atom_pos.append( atom_pos_np[chain] )
        chains_atom_mask.append( atom_mask[chain] )
        chains_res_idx.append( np.arange(len(chain), dtype=np.int32) )
        chains_bfactor.append( bfactor_np[chain] )

    # Special cases: all same letter
    if seqs_is_all_same:
        s = seqs[0][0]
        mappings = {
            "A": 25,
            "C": 26,
            "G": 27,
            "U": 28,
            "T": 24,
        }
        s_idx = mappings[s]
        chains_res_type = [
             np.asarray([s_idx] * len(chain_atom_pos), dtype=np.int32) for chain_atom_pos in chains_atom_pos
        ]

    # To all-atom structure
    if args.affines is not None and args.torsions is not None:
        print("# All-atom reconstruction using selected torsions")
        pred_affines = np.load(args.affines)
        pred_torsions = np.load(args.torsions)

        new_chains_atom_pos = []
        new_chains_atom_mask = []
        for k, chain in enumerate(fix_chains_output.chains):
            atom_pos, atom_mask = affines_and_torsions_to_all_atom(
                pred_affines[chain],
                pred_torsions[chain],
                chains_res_type[k],
            )

            atom_pos, atom_mask = atom27_to_atom23(atom_pos, atom_mask, chains_res_type[k])
            new_chains_atom_pos.append(atom_pos)
            new_chains_atom_mask.append(atom_mask)

        # Update
        chains_atom_pos = new_chains_atom_pos
        chains_atom_mask = new_chains_atom_mask

    # Output final structure 
    os.makedirs(args.output_dir, exist_ok=True)
    fout = os.path.join(args.output_dir, "denovo.cif")
    chains_atom_pos_to_pdb(
        fout,
        chains_atom_pos,
        chains_atom_mask,
        chains_res_type,
        chains_res_idx,
        chains_bfactor=chains_bfactor, 
        suffix='cif',
    )

    tend = time.time()
    print("# {:.4f} seconds".format(tend - tstart))
    print("# Done inferenec aa")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser).parse_args()
    main(args)
