import os
import sys
import math
import tempfile
import argparse
import subprocess
import numpy as np

patterns = [
    ('.', '.'), 
    ('(', ')'), ('[', ']'), ('{', '}'), ('<', '>'),
    ('A', 'a'), ('B', 'b'), ('C', 'c'), ('D', 'd'),
    ('E', 'e'), ('F', 'f'), ('G', 'h'), ('H', 'h'),
    ('I', 'i'), ('J', 'j'), ('K', 'k'), ('L', 'l'),
    ('M', 'm'), ('N', 'n'), ('O', 'o'), ('P', 'p'),
    ('Q', 'q'), ('R', 'r'), ('S', 's'), ('T', 't'),
    ('U', 'u'), ('V', 'v'), ('W', 'w'), ('X', 'x'),
    ('Y', 'y'), ('Z', 'z'),
]
patterns_left = []
patterns_right = []
patterns_left_to_idx = {}
patterns_right_to_idx = {}
patterns_idx_to_left = {}
patterns_idx_to_right = {}
for i, p in enumerate(patterns):
    l = p[0]
    r = p[1]
    patterns_left.append(l)
    patterns_right.append(r)
    patterns_left_to_idx[l] = i
    patterns_right_to_idx[r] = i
    patterns_idx_to_left[i] = l
    patterns_idx_to_right[i] = r

patterns_left_to_right = {}
for p in patterns:
    l = p[0]
    r = p[1]
    patterns_left_to_right[l] = r

patterns_right_to_left = {}
for l, r in patterns_left_to_right.items():
    patterns_right_to_left[r] = l


def get_level(ss, l):
    if ss[l[0]] != '.' or ss[l[1]] != '.':
        return 0
    for level in range(1, len(patterns)):
        score = 0
        flag = 1
        for i in range(l[0]+1, l[1]):
            if ss[i] == patterns[level][0]:
                score += 1
            elif ss[i] == patterns[level][1]:
                score -= 1
            if score < 0:
                flag = 0
                break
        if score != 0:
            flag = 0
        if flag == 1:
            return level
    return 0

def pairs_to_ss(l, pairs):
    size = l
    ss = ['.'] * l
    for l in pairs:
        level = get_level(ss, l)
        if level != 0:
            ss[l[0]] = patterns[level][0]
            ss[l[1]] = patterns[level][1]
    return "".join(ss)

def filter_pairs(pairs):
    ret = []
    idx_max = 0
    for p in pairs:
        idx_max = max(idx_max, p[0])
        idx_max = max(idx_max, p[1])
    length = idx_max + 1
    flag = [0] * length
    for p in pairs:
        flag[p[0]] = 1
        flag[p[1]] = 1

    for p in pairs:
        l, r = p
        if l - 1 >= 0 and flag[l - 1] == 0 and \
            l + 1 < length and flag[l + 1] == 0:
            continue
        if r - 1 >= 0 and flag[r - 1] == 0 and \
            r + 1 < length and flag[r + 1] == 0:
            continue
        ret.append(p)
    return ret

def ss_to_sse(ss, n_pair_min=2, verbose=False):
    sses = []
    l = len(ss)
    stacks = [[] for i in range(len(patterns))]
    i = 0
    unpaired_stack = []

    while i < l:
        ch = ss[i]

        if ch == '.':
            unpaired_stack.append((i, ch))

        if ch in patterns_left_to_idx.keys():
            idx = patterns_left_to_idx[ch]
            stacks[idx].append((i, ch))
        # is a right bracket
        else:
            k = i
            while k < l and ss[k] == ch:
                k += 1
            n = k - i
            print("# Intend to push out {} pairs".format(n))
            # push out n chs
            idx = patterns_right_to_idx[ch]
            head = []
            while n > 0:
                if not stacks[idx]:
                    raise Exception("Error invalid secondary structure -> {}".format(ss))
                top = stacks[idx][-1]
                head.append(top)
                stacks[idx].pop()
                n -= 1
            head = head[::-1]

            # head: ( ( ( ( ( (
            # tail: ) ) ) ) ) )

            # unpaired: . . . . . .
            unpaired = []
            """
            start_idx = head[0][0]
            while unpaired_stack:
                top = unpaired_stack[-1]
                if top[0] >= start_idx:
                    unpaired.append(top)
                    unpaired_stack.pop()
                else:
                    break
            """

            tail = []
            for m in range(i, k):
                tail.append((m, ss[m]))

            sse = head + tail + unpaired

            # sort sse by index
            sse.sort(key=lambda x:x[0])

            sses.append(sse)
            i = k - 1
        i += 1

    # the left stacks should be empty
    valid = True
    for i, s in enumerate(stacks):
        if i == 0:
            continue
        if verbose:
            print("# Check stack for idx {} -> {}".format(i, s))
        if s:
            raise Exception("Error there are unmatched brackets in the stack maybe you have input invalid secondary structure -> {}".format(ss))
    sses = [sse for sse in sses if len(sse) >= 2 * n_pair_min]
    print("# Split to {} SSEs".format(len(sses)))
    return sses 


def dbn_to_pairs(dbn):
    stacks = [[] for i in range(len(patterns))]
    i = 0
    l = len(dbn)

    pairs = []
    while i < l:
        ch = dbn[i]
        if ch in patterns_left_to_idx.keys():
            idx = patterns_left_to_idx[ch]
            stacks[idx].append((i, ch))
        else:
            if ch in patterns_right_to_idx.keys():
                idx = patterns_right_to_idx[ch]
                
                left = stacks[idx][-1]
                stacks[idx].pop()

                right = (i, ch)

                pairs.append((left, right))
        i += 1

    return pairs

def dbn_to_pairing_matrix(dbn):
    pairs = dbn_to_pairs(dbn)
    l = len(dbn)
    pairing_matrix = np.zeros((l, l))
    for p in pairs:
        pairing_matrix[p[0][0], p[1][0]] = 1
        pairing_matrix[p[1][0], p[0][0]] = 1
    return pairing_matrix

def print_sse(sse):
    idx_str = "# "
    bracket_str = "# "
    for site in sse:
        print(site)

    """
        idx_str += "{} ".format(site[0])
        bracket_str += "{}".format(site[1])
    print(idx_str)
    print(bracket_str)
    """


def get_pair_indices_numpy(pair_matrix: np.ndarray, threshold=0.5):
    i, j = np.triu_indices(pair_matrix.shape[0], k=1)
    mask = pair_matrix[i, j] > threshold
    return list(zip(i[mask], j[mask]))




if __name__ == '__main__':
    dbn = "<<.<<.<<<<<<<<.<<<<<..<<<<......>>>>>>>>>>>>>>>>>>>>>.<<<<<<<<.<<.<<<<<...........>>>>>...>>.>>>>>>>><<<<<<<<<<<<....>>>>>>>>>>>>"
    """
    sse = ss_to_sse(dbn)
    print_sse(sse)
    """

    pairs = dbn_to_pairs(dbn)

    pairing_matrix = dbn_to_pairing_matrix(dbn)
    print(pairing_matrix[0, 52])


