"""Apply the kill test written in EXPERIMENTS.md to whatever has run. python3 kill.py [--tier 5m]

Prints each test, its number and its failure number. A test with missing runs says so and counts as not run. Any failure at a tier means this tier can't study why planning fails, and the next step is a larger tier, not a design.
"""
import argparse
import numpy as np
from beat import load, tuned, items, named_faults, FAULT_SLOT
from world import TASKS

p = argparse.ArgumentParser(); p.add_argument("--tier", default="5m"); a = p.parse_args(); rows = load(); faults = load("results_tag.jsonl"); S = {t: tuned(rows, a.tier, t, "steps")[0] for t in TASKS}; L = {t: S[t]["lewm"] for t in TASKS if "lewm" in S[t]}
if not L: raise SystemExit(f"no lewm rows at {a.tier}: nothing to judge. Commands are in EXPERIMENTS.md.")
def line(n, x): print(f"{n}: {x}")
oo = {t: L[t]["scores"]["success.oooo"] for t in L}; line("K0 everything oracle succeeds", " ".join(f"{t} {v:.2f}" for t, v in oo.items()) + (" FAILS under .9: the task or the horizon is wrong" if min(oo.values()) < 0.9 else " passes"))
ll = {t: L[t]["scores"]["success.llll"] for t in L}; mid = [t for t, v in ll.items() if 0.1 <= v <= 0.9]; line("K1 everything learned sits between .1 and .9 on at least 2 tasks", " ".join(f"{t} {v:.2f}" for t, v in ll.items()) + (" FAILS: no failures to attribute, or a model that can't plan at all" if len(mid) < 2 else " passes"))
fl = {t: L[t]["scores"]["noise.flip"] for t in L}; gain = {t: max(L[t]["scores"][f"shap.{s}"] for s in "EPCS") for t in L}; line("K2 top credit is above planner-noise flips", " ".join(f"{t} {gain[t]:.2f}/{fl[t]:.2f}" for t in L) + (" FAILS on every task" if all(gain[t] <= fl[t] for t in L) else " passes"))
got = {t: named_faults(L[t], faults, a.tier, t) for t in L}; flat = [(t, f, v) for t in got for f, v in got[t].items()]
if not flat: line("K3 planted faults are named", "not run")
else: line("K3 planted faults are named", f"{sum(v for *_, v in flat)} of {len(flat)}, wrong: {[(t, f) for t, f, v in flat if not v]}" + (" FAILS under 90%: the attribution can't be trusted at this size" if sum(v for *_, v in flat) / len(flat) < 0.9 else " passes"))
if "rooms" in L:
    s = L["rooms"]["scores"]; g, h = s["success.llnl.o75"] - s["success.llll.o75"], s["success.loll.o75"] - s["success.llll.o75"]; line("K4 the published disagreement shows up on rooms at the far offset", f"graph cost {g:+.2f}, true predictor {h:+.2f}" + (" FAILS: neither moves, so the toy lacks the failure the papers argue about" if g <= 0 and h <= 0 else " passes"))
else: line("K4", "rooms not run")
if "small" in S and "lewmidm" in S["small"] and "lewm" in S["small"]:
    x, y = S["small"]["lewmidm"]["scores"]["success.llll"], S["small"]["lewm"]["scores"]["success.llll"]; line("K5 inverse dynamics, a published fix, moves the small object", f"inverse dynamics on small {y:.2f} to {x:.2f}" + (" FAILS" if x <= y else " passes"))
else: line("K5", "lewmidm on small not run")
for c in ("wide", "broken"):
    if "push" in S and c in S["push"] and "lewm" in S["push"]:
        d = items(S["push"][c])["success.llll"] - items(S["push"]["lewm"])["success.llll"]; z = d.mean() / max(d.std() / len(d) ** 0.5, 1e-9); line(f"K6 control {c}", f"{d.mean():+.3f}, z {z:+.1f}" + ((" FAILS: the broken control gets credit" if z >= 2 else " passes") if c == "broken" else (" passes: more weights are visible to this test" if z >= 2 else " FAILS: more weights earn nothing, so the test can't see model quality")))
    else: line(f"K6 control {c}", "not run")
