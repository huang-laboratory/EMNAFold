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


def ss_to_sse(ss, n_pair_min=2, verbose=False):
    sses = []
    l = len(ss)
    stacks = [[] for i in range(len(patterns))]
    i = 0

    while i < l:
        ch = ss[i]

        if ch == '&':
            i += 1
            continue

        if ch in patterns_left_to_idx.keys():
            idx = patterns_left_to_idx[ch]
            stacks[idx].append((i, ch))
        # is a right bracket
        else:
            k = i
            while k < l and ss[k] == ch:
                k += 1
            n = k - i
            #print("# Intend to push out {} pairs".format(n))

            # push out n chs
            idx = patterns_right_to_idx[ch]
            head = []
            while n > 0 and stacks[idx]:
                """
                if not stacks[idx]:
                    raise Exception("Error invalid secondary structure -> {}".format(ss))
                """

                top = stacks[idx][-1]
                head.append(top)
                stacks[idx].pop()
                n -= 1
            head = head[::-1]

            if head:
                # head: ( ( ( ( ( (
                # tail: ) ) ) ) ) )

                tail = []
                for m in range(i, k):
                    tail.append((m, ss[m]))

                sse = head + tail

                # sort sse by index
                sse.sort(key=lambda x:x[0])

                sses.append(sse)
                i = k - 1
        i += 1

    # the left stacks should be empty
    """
    valid = True
    for i, s in enumerate(stacks):
        if i == 0:
            continue
        if verbose:
            print("# Check stack for idx {} -> {}".format(i, s))
        if s:
            raise Exception("Error there are unmatched brackets in the stack maybe you have input invalid secondary structure -> {}".format(ss))
    """

    sses = [sse for sse in sses if len(sse) >= 2 * n_pair_min]
    #print("# Split to {} SSEs".format(len(sses)))
    return sses



def parse_ss_tree(dbn):
    dbn_list = [x for x in dbn]

    final_sses = []
    while True:
        sses = ss_to_sse(dbn_list)
        # find the largest one

        if len(sses) == 0:
            break

        sse = sses[0]

        filtered_sse = []
        for i in range(sse[0][0], sse[-1][0] + 1):
            if dbn_list[i] != "&":
                dbn_list[i] = "&"
                filtered_sse.append(i)

        final_sses.append(filtered_sse)

        #print(filtered_sse)
        #dbn = "".join(dbn_list)
        #print(dbn)

    final_loop_sse = []
    for i in range(len(dbn_list)):
        if dbn_list[i] != "&":
            final_loop_sse.append(i)

    final_sses.append(final_loop_sse)
    return final_sses

if __name__ == '__main__':
    dbn = "<<.<<.<<<<<<<<.<<<<<..<<<<...[[[[[...>>>>>>>>>>>>>>>>>>>>>.<<<<<<<<.<<.<<<<<.....]]]]]......>>>>>...>>.>>>>>>>><<<<<<<<<<<<....>>>>>>>>>>>>"
    dbn = ".........<<<<<<<<<<..(((.>>>>>>>>>>.<<<<<<<<<<<......<<<<<<.>>>>>.>>>>>>..<<<<<.<....<<<<<<...<<<<<<.....<<<.<<<<.<<...<<<<<<<<<....>>>>>>>>>..<<..)))..>>.....>>......>>>>>>>....>>>>>>..>>.>>>><<...<<<<...<<<<<<<<<...>>>>>>>>>..>>>>...>>.....(((((...>.>>>>>.<<<<<<<.....>>>>>>>........)))))...((.<<<<<....>>>>><<<<<<<....<<<<>>>>>>>>>>....>>>>>>>.<<<<<<<<.<<<<<......>>>>>.>>>>>>>>..)).."
    sses = parse_ss_tree(dbn)

    for sse in sses:
        print(sse)

