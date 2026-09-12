"""Fit research/simulate.py's Schedule to real meeting turn statistics (PLAN.md §8 job 5).

Reads pyannote AMI reference RTTMs (train+dev split, never test) and reports:
turn length quantiles, inter-turn gap distribution (negative = overlap at
handover), overlap fraction of speech time, same-speaker continuation rate,
short-turn (backchannel-like, < 0.8 s) rate. Prints a Schedule(...) line to
paste into simulate.py.

    python research/fit_turns.py bench/ami/setup/only_words/rttms/train bench/ami/setup/only_words/rttms/dev
"""
import glob
import sys

import numpy as np


def load(path):
    segs = []
    for line in open(path):
        p = line.split()
        if p and p[0] == "SPEAKER":
            segs.append((float(p[3]), float(p[3]) + float(p[4]), p[7]))
    return sorted(segs)


def merge_same_speaker(segs, gap=0.3):
    """Reference RTTMs split on every pause; merge a speaker's segments separated by < gap."""
    out = []
    for s in segs:
        if out and out[-1][2] == s[2] and s[0] - out[-1][1] < gap:
            out[-1] = (out[-1][0], max(out[-1][1], s[1]), s[2])
        else:
            out.append(s)
    return sorted(out)


def stats(files):
    lens, gaps, cont, short, ov_frac = [], [], [], 0, []
    for f in files:
        segs = merge_same_speaker(load(f))
        if not segs:
            continue
        # overlap fraction of speech time at 80 ms resolution
        T = int(max(e for _, e, _ in segs) / 0.08) + 1
        cnt = np.zeros(T, int)
        for s, e, _ in segs:
            cnt[int(s / 0.08):int(e / 0.08) + 1] += 1
        ov_frac.append((cnt > 1).sum() / max((cnt > 0).sum(), 1))
        turns = sorted(segs)
        for i, (s, e, spk) in enumerate(turns):
            lens.append(e - s); short += (e - s) < 0.8
            if i + 1 < len(turns):
                ns, ne, nspk = turns[i + 1]
                gaps.append(ns - e); cont.append(nspk == spk)
    lens, gaps = np.array(lens), np.array(gaps)
    q = lambda a, p: float(np.quantile(a, p))
    print(f"files {len(files)}  turns {len(lens)}")
    print(f"turn length s: p10 {q(lens,.1):.2f} p50 {q(lens,.5):.2f} p90 {q(lens,.9):.2f}  short(<0.8s) {short/len(lens):.2f}")
    print(f"gap s (neg=overlap): p10 {q(gaps,.1):.2f} p50 {q(gaps,.5):.2f} p90 {q(gaps,.9):.2f}  frac gap<0 {np.mean(gaps<0):.2f}")
    print(f"same-speaker continuation {np.mean(cont):.2f}   overlap fraction of speech time {np.mean(ov_frac):.3f}")
    lo, hi = q(lens, .1), q(lens, .9)
    print("\nSchedule(turn_s=(%.2f, %.2f), gap_s=(%.2f, %.2f), overlap_p=%.2f, backchannel_p=%.2f, "
          "backchannel_s=(0.2, 0.8), self_continue_p=%.2f)" % (
              max(lo, 0.3), hi, q(gaps, .1), q(gaps, .9), np.mean(gaps < 0), short / len(lens), np.mean(cont)))


if __name__ == "__main__":
    files = [f for d in sys.argv[1:] for f in sorted(glob.glob(f"{d}/*.rttm"))]
    stats(files)
