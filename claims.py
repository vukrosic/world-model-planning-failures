"""Does this environment reproduce what the papers report, in direction? python3 claims.py TIER

Published numbers come from other tasks, simulators and sizes, so only the direction is compared, and each row says what it would mean. A gain holds when it has the published sign with z of at least 2, episode by episode, and a tie doesn't hold. "About equal" holds when the gap and 2 standard errors stay within .1. Rows at a tier with no run say so. A mismatch needs a targeted run that finds the reason, before any design runs. Each z is across episodes of one trained model, so read it beside kill.py's null floor.
"""
import argparse
import numpy as np
from beat import load, tuned, items

p = argparse.ArgumentParser(); p.add_argument("tier"); a = p.parse_args(); rows = load()
def R(task, m="lewm"): return tuned(rows, a.tier, task, "steps")[0].get(m)
def G(rx, kx, ry, ky, off=None):
    """(mean gap kx - ky, its standard error), optionally at one offset. Rows on the same task and world share episodes."""
    A, C = items(rx), items(ry); d = A[kx] - C[ky]
    if off: d = d[A["offset"] == off]
    return float(d.mean()), float(d.std() / max(len(d), 1) ** 0.5)
def stuck(*v): return bool(v) and all(x <= 0.02 or x >= 0.98 for x in v)
def line(name, pub, x, y, gap, se, want="gain"):
    ok = gap > 0 and gap >= 2 * se if want == "gain" else gap < 0 and -gap >= 2 * se if want == "loss" else abs(gap) + 2 * se <= 0.1
    print(f"| {name} | {pub} | {x:.2f} to {y:.2f}, gap {gap:+.2f} +/- {se:.2f} | {'cannot tell: the score sits at 0 or 1' if stuck(x, y) else 'holds' if ok else 'does not hold'} |")
print(f"tier {a.tier}\n| claim | published | here | direction |\n|---|---|---|---|")
LL, TP, NG = "success.llll", "success.loll", "success.llnl"
for task, pub, want in (("rooms", "IMWM: true predictor 87.7 to 85.5 on Two-Room, about equal", "equal"), ("push", "IMWM: true predictor 66.2 to 76.8 on OGBench-Cube", "gain")):
    r = R(task)
    if r: line(f"True predictor under the same search on {task}", pub, r["scores"][LL], r["scores"][TP], *G(r, TP, r, LL), want)
    else: print(f"| True predictor on {task} | {pub} | not run | |")
for task in ("rooms", "push"):
    r = R(task); s = r["scores"] if r else {}
    if "diag.zero_when_failed" in s: print(f"| Failed episodes with no sample reaching the goal on {task} | IMWM: 98.9% and 91.4% | {s['diag.zero_when_failed']:.2f} | {'holds' if s['diag.zero_when_failed'] >= 0.9 else 'does not hold'} |")
    else: print(f"| No sample reaches the goal on {task} | IMWM: 98.9% and 91.4% | not run | |")
for task, pub in (("rooms", "NGC: farthest offset 27% to 94.6% on TwoRoom"), ("reach", "NGC: farthest offset 81% to 99.4% on Reacher")):
    r = R(task)
    if r: line(f"Graph cost beats latent distance at the farthest offset on {task}", pub, r["scores"]["success.llll.o75"], r["scores"]["success.llnl.o75"], *G(r, NG, r, LL, 75))
    else: print(f"| Graph cost on {task} | {pub} | not run | |")
s, t = R("small"), R("small", "lewmidm")
if s and t: line("Inverse dynamics beats LeWM on the small object", "SLIM: .003 to .35", s["scores"][LL], t["scores"][LL], *G(t, LL, s, LL))
else: print("| Inverse dynamics on the small object | SLIM: .003 to .35 | not run | |")
s, t = R("push"), R("wind")
if s and t:
    x, y = items(s)[LL], items(t)[LL]; g = float(y.mean() - x.mean()); se = float((x.var() / len(x) + y.var() / len(y)) ** 0.5); line("A world effect lowers LeWM, across different episodes", "DWM: 94.0 to 32 on Push-T", float(x.mean()), float(y.mean()), g, se, "loss")
else: print("| A world effect lowers LeWM | DWM: 94.0 to 32 on Push-T | not run | |")
s, t = R("push"), R("push", "pldm")
if s and t: line("LeWM beats PLDM on the push task", "LeWM: 18 points on Push-T", t["scores"][LL], s["scores"][LL], *G(s, LL, t, LL))
else: print("| LeWM beats PLDM on push | LeWM: 18 points on Push-T | not run | |")
for task in ("push", "rooms"):
    r = R(task)
    if r: I = items(r); n, f = I[LL][I["offset"] == 25], I[LL][I["offset"] == 75]; line(f"Success falls from the nearest offset to the farthest on {task}", "NGC and LeWM's offset tables", float(n.mean()), float(f.mean()), float(f.mean() - n.mean()), float((n.var() / len(n) + f.var() / len(f)) ** 0.5), "loss")
