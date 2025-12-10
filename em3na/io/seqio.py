from Bio import pairwise2
from Bio.Seq import Seq
import warnings
warnings.filterwarnings("ignore")
import numpy as np

def readlines(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if line.strip()]
    return lines

def read_lines(filename):
    lines = readlines(filename)
    return lines

def read_fasta(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    seqs = []
    seq = ""
    for line in lines:
        if line.startswith('>'):
            if len(seq) > 0:
                seqs.append(seq)
            seq = ""
            continue
        seq += line.strip()
    if len(seq) > 0:
        seqs.append(seq)
    return seqs

def is_all_acgut(seq):
    for ch in seq:
        if ch not in ["A", "C", "G", "U", "T"]:
            return False
    return True

def is_all_same(seq, character):
    assert len(seq) > 0
    for ch in seq:
        if ch != character:
            return False
    return True

def filter_non_acgut(seq):
    new_seq = []
    for ch in seq:
        if ch in ["A", "C", "G", "U", "T"]:
            new_seq.append(ch)
    new_seq = "".join(new_seq)
    return new_seq

def write_lines_to_file(lines, filename):
    with open(filename, 'w') as f:
        for line in lines:
            f.write(line.strip("\n") + "\n")

def write_seqs_to_file(seqs, filename):
    lines = []
    for k, seq in enumerate(seqs):
        lines.append(f">Seq_{k}")
        lines.append(seq.strip())
    write_lines_to_file(lines, filename)

def read_secstr(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    seqs = []
    seq = ""
    for line in lines:
        if line.startswith('>'):
            if len(seq) > 0:
                seqs.append(seq)
            seq = ""
            continue
        seq += line.strip()
    if len(seq) > 0:
        seqs.append(seq)
    return seqs
    '''
    with open(filename, 'r') as f:
        lines = f.readlines()
    secs = []
    for line in lines:
        if len(line.strip()) > 0:
            secs.append(line.strip())
    return secs
    '''

def toupper(seq):
    return seq.upper()

def tolower(seq):
    return seq.lower()

def is_na(seq):
    nucs = ['A', 'G', 'C', 'U', 'T', 'I', 'N', 'X', 'a', 'g', 'c', 'u', 't', 'i', 'n', 'x']
    for s in seq:
        if s not in nucs:
            return False
    return True

def is_dna(seq):
    return is_na(seq) and ('T' in seq or 't' in seq)

def is_rna(seq):
    return is_na(seq) and 'T' not in seq and 't' not in seq

def align(a, b, mode='global', match=2, mismatch=-1, gap_open=-0.5, gap_extend=-0.1, n_classes=4, setting=1, **kwargs):
    seq1 = Seq(a)
    seq2 = Seq(b)

    # Get match dict
    if setting == 0:
        match_dict = {
            ('A', 'A'): 2, ('A', 'G'): 1, ('A', 'C'):-1, ('A', 'U'):-1, ('A', 'T'):-1,
            ('G', 'A'): 1, ('G', 'G'): 2, ('G', 'C'):-1, ('G', 'U'):-1, ('G', 'T'):-1,
            ('C', 'A'):-1, ('C', 'G'):-1, ('C', 'C'): 2, ('C', 'U'): 1, ('C', 'T'): 1,
            ('U', 'A'):-1, ('U', 'G'):-1, ('U', 'C'): 1, ('U', 'U'): 2, ('U', 'T'): 1,
            ('T', 'A'):-1, ('T', 'G'):-1, ('T', 'C'): 1, ('T', 'U'): 1, ('T', 'T'): 2,
        }
    else:
        aa_match_matrix = np.array(
            [[0.20466549, 0.13293310, 0.07639524, 0.08726946],
             [0.12993139, 0.23787625, 0.07880677, 0.06438896],
             [0.07319305, 0.07168299, 0.22714286, 0.11792232],
             [0.08726946, 0.05300044, 0.17080019, 0.18721251]]
        )
        aa_match_matrix = (aa_match_matrix - 0.05) * 10

        names = ['A', 'G', 'C', 'U', 'T']
        match_dict = dict()
        for i in range(4):
            for k in range(4):
                match_dict[(names[i], names[k])] = aa_match_matrix[i][k]

    if mode == 'global':
        alignments = pairwise2.align.globalds(seq1, seq2, match_dict, gap_open, gap_extend, **kwargs)
    elif mode == 'local':
        alignments =  pairwise2.align.localds(seq1, seq2, match_dict, gap_open, gap_extend, **kwargs)
    else:
        raise ValueError
    return alignments


def get_seq_align_among_candidates(seq1, seqs, **kwargs):
    # Determine which sequence best fits the segment
    score0 = -1e6
    seqA0 = None
    seqB0 = None
    k0 = None

    for k, seq2 in enumerate(seqs):
        # Target sequence is >= query sequence

        alignment = align(seq1, seq2, gap_open=-1.0, gap_extend=-0.5, **kwargs)[0]
        seqA, seqB, score, start, end = alignment[:5]

        if score > score0:
            score0 = score
            seqA0 = seqA[start:end+1]
            seqB0 = seqB[start:end+1]
            k0 = k

    seqA = seqA0
    seqB = seqB0

    # In case that given sequence is invalid
    if seqA is None or \
       seqB is None:
        seqA = seq1
        seqB = seq1

    seqA, seqB = remove_bar(seqA, seqB)

    # In case that seqB is shorter
    seqB = seqB + 'U'*(len(seq1)-len(seqB))
    return seqA, seqB, score0, k0

def remove_bar(seq1, seq2, sec2=None):
    assert len(seq1) == len(seq2)
    if sec2 is not None:
        assert len(seq2) == len(sec2)

    # Remove bar both terminus in seq1
    n = len(seq1)
    ret_seq1 = []
    ret_seq2 = []
    ret_sec2 = []
    for i in range(n):
        if seq1[i] != '-':
            ret_seq1.append(seq1[i])
            ret_seq2.append(seq2[i])
            if sec2 is not None:
                ret_sec2.append(sec2[i])

    # s1 does not have '-'
    # s2 may have '-'
    n = len(ret_seq2)
    for i in range(n):
        if ret_seq2[i] == '-':
            ret_seq2[i] = ret_seq1[i]
            if sec2 is not None:
                ret_sec2[i] = '.'

    ret_seq1 = "".join(ret_seq1)
    ret_seq2 = "".join(ret_seq2)
    ret_sec2 = "".join(ret_sec2)
    if sec2 is not None:
        return ret_seq1, ret_seq2, ret_sec2
    else:
        return ret_seq1, ret_seq2

if __name__ == '__main__':
    seqA, seqB, score, k = get_seq_align_among_candidates(
        "AAGUGCUGG",
        ["AAGGGUUGC", "AAGGAGGAGC"]
    )
    print(k, score)
    print(seqA)
    print(seqB)
