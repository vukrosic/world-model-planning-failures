"""Apply the kill test written in EXPERIMENTS.md to whatever has run. python3 kill.py [--tier 5m]

Prints each test, its number and its failure number. A test with missing runs says so and counts as not judged. Any failure at a tier means this tier can't study why planning fails, and the next step is a larger tier, not a design.
N0 runs the null test first: lewm against itself on fresh worlds. Passing a gain test takes a paired z, at least 2 fresh worlds and a fresh-world gap above that floor.
"""
import argparse
import beat
from world import TASKS

LL = beat.LL


def report(rows, faults, tier="5m", say=print):
    """Returns {test: status}, where status is passes, FAILS or not judged."""
    S = {t: beat.tuned(rows, tier, t, "steps")[0] for t in TASKS}; L = {t: S[t]["lewm"] for t in TASKS if "lewm" in S[t]}; out = {}

    def line(name, status, text): out[name] = status; say(f"{name}: {status}. {text}")
    if not L: say(f"no lewm rows at {tier}: nothing to judge. Commands are in EXPERIMENTS.md."); return out
    fl = {t: beat.null_floor(rows, tier, t, "steps") for t in L}; line("N0 null test", "passes" if min(n for _, n in fl.values()) >= 3 else "not judged", "lewm against lewm on fresh worlds, largest success gap: " + " ".join(f"{t} {v:.3f} ({n})" for t, (v, n) in fl.items()) + ". A world changes data, weights and episodes, and each gain below must clear this floor")
    oo = {t: L[t]["scores"]["success.oooo"] for t in L}; line("K0 everything oracle succeeds", "FAILS" if min(oo.values()) < 0.9 else "passes", " ".join(f"{t} {v:.2f}" for t, v in oo.items()) + ". Under .9 means the task or the horizon is wrong")

    def above(t):
        """llll beats holding still and random plans by z 2 on this task."""
        I = beat.items(L[t]); return all(beat.zgap(I[LL], I[f"success.{f}"]) >= 2 for f in ("idle", "random"))
    mid = [t for t in L if 0.1 <= L[t]["scores"][LL] <= 0.9 and above(t)]; line("K1 everything learned sits between .1 and .9 and above the no-model floor on at least 2 tasks", "FAILS" if len(mid) < 2 else "passes", " ".join(f"{t} {L[t]['scores'][LL]:.2f} (still {L[t]['scores']['success.idle']:.2f}, random {L[t]['scores']['success.random']:.2f})" for t in L) + f". These count: {', '.join(mid) or 'none'}")
    top = {t: max(L[t]["scores"][f"shap.{s}"] for s in "EPCS") for t in L}; nz = {t: L[t]["scores"]["noise.credit"] for t in L}; line("K2 top credit is above planner noise", "FAILS" if all(top[t] <= 2 * nz[t] for t in L) else "passes", " ".join(f"{t} {top[t]:.2f}/{2 * nz[t]:.3f}" for t in L) + ". Both sides are in success units: the credit and twice its standard deviation from planner noise")
    got = {t: beat.named_faults(L[t], faults, tier, t) for t in L}; flat = [(t, f, v) for t in got for f, v in got[t].items()]
    if not flat: line("K3 planted faults are named", "not judged", "no fault rows at the healthy rows' settings")
    else: right = sum(v for *_, v in flat); line("K3 planted faults are named", "FAILS" if right / len(flat) < 0.9 else "passes", f"{right} of {len(flat)}, wrong: {[(t, f) for t, f, v in flat if not v]}. Under 90% means the attribution can't be trusted at this size")
    if "rooms" in L:
        r = L["rooms"]; I = beat.items(r); far = I["offset"] == 75; g, h = I["success.llnl"][far] - I["success.llll"][far], I["success.loll"][far] - I["success.llll"][far]; se = lambda d: max(d.std() / len(d) ** 0.5, 1e-9); zg, zh = float(g.mean() / se(g)), float((g - h).mean() / se(g - h)); c = r["scores"].get("ngc.connected", 1)
        status = "not judged" if c < 0.9 else "passes" if zg >= 2 and zh >= 2 else "FAILS"; line("K4 the published disagreement shows on rooms at the far offset", status, f"graph cost gains {g.mean():+.2f}, z {zg:+.1f}, true predictor gains {h.mean():+.2f}, graph over predictor z {zh:+.1f}, graph connected {c:.2f}. Published: the graph cost gains much and the true predictor about nothing. Failure means the toy lacks the failure the papers argue about")
    else: line("K4 the published disagreement shows on rooms at the far offset", "not judged", "rooms not run")
    if "small" in S and "lewmidm" in S["small"] and "lewm" in S["small"]: st, t = beat.verdict(rows, tier, "small", "steps", S["small"]["lewmidm"], S["small"]["lewm"]); line("K5 inverse dynamics, a published fix, moves the small object", st, t)
    else: line("K5 inverse dynamics, a published fix, moves the small object", "not judged", "lewmidm on small not run")
    b = [(t, S[t]) for t in S if "broken" in S[t] and "lewm" in S[t]]
    if b:
        res = [(t, beat.zgap(beat.items(T["broken"])[LL], beat.items(T["broken"])["success.random"]), beat.zgap(beat.items(T["broken"])[LL], beat.items(T["lewm"])[LL])) for t, T in b]; line("K6 the broken control earns nothing", "FAILS" if any(zr >= 2 or zl >= 2 for _, zr, zl in res) else "passes", " ".join(f"{t}: over random z {zr:+.1f}, over lewm z {zl:+.1f}" for t, zr, zl in res) + ". A blank image can only match random plans")
    else: line("K6 the broken control earns nothing", "not judged", "broken not run")
    if "push" in S and "wide" in S["push"] and "lewm" in S["push"]: st, t = beat.verdict(rows, tier, "push", "steps", S["push"]["wide"], S["push"]["lewm"]); line("K7 the test can see more weights", st, t + ". Failure means the test can't see model quality")
    else: line("K7 the test can see more weights", "not judged", "wide not run")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--tier", default="5m"); a = p.parse_args(); report(beat.load(), beat.load("results_tag.jsonl"), a.tier)
