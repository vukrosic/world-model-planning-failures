"""Which slot failed. Four slots, each learned or oracle, make 16 cells on the same episodes.

Slots: E encoder (pixels or true state), P predictor (learned or the true task), C cost (latent distance or the true cost), S search (CEM at its budget or the oracle search).
A cell is four letters in that order: llll is everything learned, oooo is everything oracle.
Shapley credit of a slot, per episode: the mean gain from making that slot oracle over every order of making slots oracle. The credits add up to v(oooo) - v(llll) exactly, so interactions are split and not lost, and each interaction is printed separately.

Two limits on reading these. E is not an encoder alone: E=o reads the true state and also uses the predictor, probe and graph trained on it, because a predictor trained on pixel latents can't read another latent. So E's credit means "what the model reads, with the dynamics learned on it". And with P and C both oracle the encoder is unused, so a third of E's Shapley weight (1/12 + 1/4) is zero by construction. solo() reads each slot from llll, where every slot has its full weight, and beat.py needs it before it calls a slot.
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


def solo(v):
    """The gain from making only this slot oracle, per episode."""
    return {s: v[cell({s})] - v[LL] for s in SLOTS}


NOISE_WEIGHT = 1 / 6  # a credit adds 16 cells with Shapley weights whose squares sum to 1/3. A cell's mean has noise variance flip/(2n), because two runs differ by flip/n. So a credit's noise variance is flip/(6n), assuming llll's flip share holds in every cell


def noise_sd(flip, n):
    """The standard deviation of a credit from planner noise alone, in success units, so it can be compared with a credit."""
    return (NOISE_WEIGHT * flip / max(n, 1)) ** 0.5


def labels(v):
    """For each failed episode of llll: one slot's swap fixes it, it takes several swaps, or it fails with everything oracle."""
    f = v[LL] < 0.5; n = max(int(f.sum()), 1); out = {f"single.{s}": float((f & (v[cell({s})] >= 0.5)).sum() / n) for s in SLOTS}; anyone = np.any([v[cell({s})] >= 0.5 for s in SLOTS], 0)
    out["several"] = float((f & ~anyone & (v[OO] >= 0.5)).sum() / n); out["none"] = float((f & ~anyone & (v[OO] < 0.5)).sum() / n); out["failed"] = float(f.mean()); return out


def interactions(v):
    """Second-order interaction per pair: the extra gain from making both oracle, beyond each alone."""
    return {a + b: v[cell({a, b})] + v[LL] - v[cell({a})] - v[cell({b})] for a, b in itertools.combinations(SLOTS, 2)}
