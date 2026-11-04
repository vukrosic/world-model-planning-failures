"""Does this environment reproduce what the papers report, in direction? python3 claims.py TIER

Published numbers come from other tasks, simulators and sizes, so only the direction is compared, and each row says what it would mean. Rows at a tier with no run say so. A mismatch needs a targeted run that finds the reason, before any design runs.
"""
import argparse
from beat import load, tuned

p = argparse.ArgumentParser(); p.add_argument("tier"); a = p.parse_args(); rows = load()
def S(task, m="lewm"): return tuned(rows, a.tier, task, "steps")[0].get(m, {}).get("scores")
def line(name, pub, got, ok, *vals): print(f"| {name} | {pub} | {got} | {'cannot tell: the score sits at 0 or 1' if vals and all(v <= 0.02 or v >= 0.98 for v in vals) else 'holds' if ok else 'does not hold'} |")
print(f"tier {a.tier}\n| claim | published | here | direction |\n|---|---|---|---|")
for task, pub, want in (("rooms", "IMWM: true predictor 87.7 to 85.5 on Two-Room", lambda d: abs(d) < 0.1), ("push", "IMWM: true predictor 66.2 to 76.8 on OGBench-Cube", lambda d: d > 0)):
    s = S(task)
    if s: d = s["success.loll"] - s["success.llll"]; line(f"True predictor under the same search on {task}", pub, f"{s['success.llll']:.2f} to {s['success.loll']:.2f}", want(d), s["success.llll"])
    else: print(f"| True predictor on {task} | {pub} | not run | |")
for task in ("rooms", "push"):
    s = S(task)
    if s and "diag.zero_when_failed" in s: line(f"Failed episodes with no sample reaching the goal on {task}", "IMWM: 98.9% and 91.4%", f"{s['diag.zero_when_failed']:.2f}", s["diag.zero_when_failed"] >= 0.9)
    else: print(f"| No sample reaches the goal on {task} | IMWM: 98.9% and 91.4% | not run | |")
for task, pub in (("rooms", "NGC: farthest offset 27% to 94.6% on TwoRoom"), ("reach", "NGC: farthest offset 81% to 99.4% on Reacher")):
    s = S(task)
    if s: line(f"Graph cost beats latent distance at the farthest offset on {task}", pub, f"{s['success.llll.o75']:.2f} to {s['success.llnl.o75']:.2f}", s["success.llnl.o75"] > s["success.llll.o75"], s["success.llnl.o75"], s["success.llll.o75"])
    else: print(f"| Graph cost on {task} | {pub} | not run | |")
s, t = S("small"), S("small", "lewmidm")
if s and t: line("Inverse dynamics beats LeWM on the small object", "SLIM: .003 to .35", f"{s['success.llll']:.2f} to {t['success.llll']:.2f}", t["success.llll"] > s["success.llll"], s["success.llll"], t["success.llll"])
else: print("| Inverse dynamics on the small object | SLIM: .003 to .35 | not run | |")
s, t = S("push"), S("wind")
if s and t: line("A world effect lowers LeWM", "DWM: 94.0 to 32 on Push-T", f"{s['success.llll']:.2f} to {t['success.llll']:.2f}", t["success.llll"] < s["success.llll"], s["success.llll"], t["success.llll"])
else: print("| A world effect lowers LeWM | DWM: 94.0 to 32 on Push-T | not run | |")
s, t = S("push"), S("push", "pldm")
if s and t: line("LeWM beats PLDM on the push task", "LeWM: 18 points on Push-T", f"{t['success.llll']:.2f} to {s['success.llll']:.2f}", s["success.llll"] > t["success.llll"], s["success.llll"], t["success.llll"])
else: print("| LeWM beats PLDM on push | LeWM: 18 points on Push-T | not run | |")
for task in ("push", "rooms"):
    s = S(task)
    if s: line(f"Success falls with the goal's offset on {task}", "NGC and LeWM's offset tables", " ".join(f"{s[f'success.llll.o{o}']:.2f}" for o in (25, 50, 75)), s["success.llll.o25"] >= s["success.llll.o75"], s["success.llll.o25"], s["success.llll.o75"])
