import numpy as np
from em3na.utils.hmm.match_to_sequence import MatchToSequence
from em3na.utils.residue_constants import restype_1_to_index

# Fill unmatched gaps
def fill_gaps(arr, upper_bound):
    mask = np.zeros_like(arr, dtype=bool) # label which one is updated

    lower_bound = 1
    length = len(arr)
    # scan from left to right until meets the first number not -1
    left_start = np.where(arr != -1)[0][0]
    for i in range(left_start - 1, -1, -1):
        next_value = arr[i + 1] - 1
        if next_value >= lower_bound:
            arr[i] = next_value
            mask[i] = True
        else:
            break

    # scan from right to left until meets the first number not -1
    right_start = np.where(arr != -1)[0][-1]
    for i in range(right_start + 1, length):
        next_value = arr[i - 1] + 1
        if next_value <= upper_bound:
            arr[i] = next_value
            mask[i] = True
        else:
            break
    return arr, mask


# Fix sequence alignment by extending at both terminus
def fix_match(best_match_output, seqs, match_score_cutoff=0.40, len_cutoff=6):
    (
        new_sequences,
        residue_idxs,
        sequence_idxs,
        key_start_matches,
        key_end_matches,
        match_scores,
        hmm_output_match_sequences,
        exists_in_sequence_mask,
        is_nucleotide_list,
    ) = ([], [], [], [], [], [], [], [], [])
    
    for i in range(len(best_match_output.residue_idxs)):
    
        # ignore bad small fragments
        if len(best_match_output.residue_idxs[i]) >= len_cutoff and \
            best_match_output.match_scores[i] >= match_score_cutoff:
    
            sequence_idx = best_match_output.sequence_idxs[i]
            # get new residue idx
            residue_idx, residue_idx_update_mask = fill_gaps(
                best_match_output.residue_idxs[i].copy(),
                upper_bound=len(seqs[sequence_idx]),
            )
            residue_idxs.append(residue_idx)
    
            # update calculated using new mask
            update_idx = np.where(residue_idx_update_mask == True)[0]
            new_sequence = best_match_output.new_sequences[i].copy()
            hmm_output_match_sequence = list(best_match_output.hmm_output_match_sequences[i])
            for k in update_idx:
                new_sequence[k] = restype_1_to_index[seqs[sequence_idx][residue_idx[k] - 1]]
                hmm_output_match_sequence[k] = seqs[sequence_idx][residue_idx[k] - 1]
            hmm_output_match_sequence = "".join(hmm_output_match_sequence)
    
            new_sequences.append(new_sequence)
            hmm_output_match_sequences.append(hmm_output_match_sequence)
    
            # simply update
            exists_in_sequence_mask.append((residue_idx != -1).astype(np.int32))
            match_scores.append((residue_idx != -1).sum() / len(residue_idx))
            key_start_matches.append(-1 if residue_idx[0] == -1 else residue_idx[0])
            key_end_matches.append(
                residue_idx[np.where(residue_idx != -1)[0][-1]]
            )
    
            sequence_idxs.append(best_match_output.sequence_idxs[i])
            is_nucleotide_list.append(best_match_output.is_nucleotide[i])
        else:
            new_sequences.append(best_match_output.new_sequences[i])
            residue_idxs.append(best_match_output.residue_idxs[i])
            sequence_idxs.append(best_match_output.sequence_idxs[i])
            key_start_matches.append(best_match_output.key_start_matches[i])
            key_end_matches.append(best_match_output.key_end_matches[i])
            match_scores.append(best_match_output.match_scores[i])
            hmm_output_match_sequences.append(best_match_output.hmm_output_match_sequences[i])
            exists_in_sequence_mask.append(best_match_output.exists_in_sequence_mask[i])
            is_nucleotide_list.append(best_match_output.is_nucleotide[i])
    
    # return
    return MatchToSequence(
        new_sequences=new_sequences,
        residue_idxs=residue_idxs,
        sequence_idxs=sequence_idxs,
        key_start_matches=np.array(key_start_matches),
        key_end_matches=np.array(key_end_matches),
        match_scores=np.array(match_scores),
        hmm_output_match_sequences=hmm_output_match_sequences,
        exists_in_sequence_mask=exists_in_sequence_mask,
        is_nucleotide=is_nucleotide_list,
    )


