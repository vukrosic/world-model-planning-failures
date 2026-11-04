"""Which slot failed. Four slots, each learned or oracle, make 16 cells on the same episodes.

Slots: E encoder (pixels or true state), P predictor (learned or the true task), C cost (latent distance or the true cost), S search (CEM at its budget or the oracle search).
A cell is four letters in that order: llll is everything learned, oooo is everything oracle.
Shapley credit of a slot, per episode: the mean gain from making that slot oracle over every order of making slots oracle. The credits add up to v(oooo) - v(llll) exactly, so interactions are split and not lost, and each interaction is printed separately.
"""
import itertools, math
import numpy as np

SLOTS = "EPCS"; CELLS = ["".join(c) for c in itertools.product("lo", repeat=4)]; LL, OO = "llll", "oooo"


def cell(on): return "".join("o" if s in on else "l" for s in SLOTS)


def shapley(v):
    """v maps every cell to a per-episode array. Returns slot -> per-episode credit."""
    out = {}
    for s in SLOTS:
        rest = [x for x in SLOTS if x != s]; t = 0
        for k in range(4):
            w = math.factorial(k) * math.factorial(3 - k) / 24
            for sub in itertools.combinations(rest, k): t = t + w * (v[cell(set(sub) | {s})] - v[cell(set(sub))])
        out[s] = t
    return out


def top_z(phi):
    """The largest slot, and its z over the second largest, paired by episode."""
    o = sorted(SLOTS, key=lambda s: -phi[s].mean()); d = phi[o[0]] - phi[o[1]]; return o[0], float(np.clip(d.mean() / max(d.std() / max(len(d), 1) ** 0.5, 1e-9), -99, 99))


def labels(v):
    """For each failed episode of llll: one slot's swap fixes it, it takes several swaps, or it fails with everything oracle."""
    f = v[LL] < 0.5; n = max(int(f.sum()), 1); out = {f"single.{s}": float((f & (v[cell({s})] >= 0.5)).sum() / n) for s in SLOTS}; anyone = np.any([v[cell({s})] >= 0.5 for s in SLOTS], 0)
    out["several"] = float((f & ~anyone & (v[OO] >= 0.5)).sum() / n); out["none"] = float((f & ~anyone & (v[OO] < 0.5)).sum() / n); out["failed"] = float(f.mean()); return out


def interactions(v):
    """Second-order interaction per pair: the extra gain from making both oracle, beyond each alone."""
    return {a + b: v[cell({a, b})] + v[LL] - v[cell({a})] - v[cell({b})] for a, b in itertools.combinations(SLOTS, 2)}
