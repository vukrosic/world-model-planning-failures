"""Two win rules, committed before any new design ran. python3 beat.py [--tiers 5m 1h] [--hidden] [--design method cell]

Rule 1: does a design beat every published world model, cost and search? A design is a method and a cell among llll, llnl, lllp or a new one. It wins only if ALL of these hold at both tiers:
1. Tuned as hard as the baselines: its lr and knob are the pair tune.dist picks, its lr is not on the edge of what it tried, and no baseline tried fewer pairs on any task.
2. Equal compute and equal steps: at both budgets, on at least 3 of the 5 tasks its success beats every baseline method at every baseline cell by at least 2 standard errors of the episode-by-episode gap.
3. Nothing lost: on no task and no budget does a baseline beat it by 2 standard errors.
4. Beyond training noise: its gap to the best baseline on the tasks it wins is at least 2 standard deviations of lewm's success across fresh worlds, with at least 3 worlds.
5. Same weights and search: test-time weights at most lewm's x 1.01, the same samples, steps and elites, and the search itself at most 1.25x lewm's llll seconds, because a cost that takes far longer is a larger search.
Fairness sentence: every baseline has the same encoder and predictor at test, because they differ only in their training loss and the training-only heads are dropped; equal compute gives each method the training steps that take lewm's time on one thread.

Rule 2: can a slot be called the bottleneck on a task? Only if ALL of these hold at the tier, and on the same slot at two tiers:
1. oooo succeeds on at least .9 and llll is between .1 and .9, so there is something to explain and the task is solvable.
2. The top slot's credit beats the second by z >= 2, episode by episode.
3. The top slot's credit is larger than noise.flip, the share of episodes that flip when only the planner's noise changes.
4. All four planted faults at full strength are named correctly at this tier, so the attribution has been shown to work at this size. A fault counts as named if its slot's credit rises most against the same row with no fault.
5. The probe reads the latent with R2 of at least .5, unless the slot called is the encoder. A weak probe understates the cost slot, because the true cost then gets wrong states.
A win or a bottleneck at these tiers earns a run on the real simulators, and no claim about them holds until that run exists. Report the number of designs tried with any win. The controls must never win: the script exits with an error if one does.
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
from model import BASELINES, CONTROLS, REF
from world import TASKS
BASE_CELLS = ("llll", "llnl", "lllp"); FAULT_SLOT = {"blind": "E", "noact": "P", "flip": "C", "tiny": "S"}; low = lambda k: k.startswith("dist")


def load(name="results.jsonl"):
    p = os.path.join(HERE, name); return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def tuned(rows, tier, task, budget, world=0, fault=""):
    """The row tune.dist picks per method, and every pair tried. tune.dist comes from tuning episodes, which are never reported."""
    best, tried = {}, {}
    for r in rows:
        if r["tier"] == tier and r["task"] == task and r["budget"] == budget and r["world"] == world and r["fault"] == fault and not r["tag"]:
            tried.setdefault(r["method"], set()).add((r["lr"], r["knob"]))
            if r["method"] not in best or r["scores"]["tune.dist"] < best[r["method"]]["scores"]["tune.dist"]: best[r["method"]] = r
    return best, tried


def items(r, hidden=False):
    n = f"{r['tier']}_{r['task']}_{r['method']}_{r['budget']}_lr{r['lr']:g}_k{r['knob']:g}_w{r['world']}" + (f"_{r['fault']}" if r["fault"] else "") + (f"_s{r['strength']:g}" if r["strength"] != 1 else "") + (f"_{r['tag']}" if r["tag"] else "") + ".npz"; return np.load(os.path.join(HERE, "runs", ("final_" if hidden else "") + n))


def zgap(a, b):
    d = a - b; return float(d.mean() / max(d.std() / len(d) ** 0.5, 1e-9))


def judge(rows, m, cell, tier, hidden=False):
    no = []; wins = {b: [] for b in ("steps", "compute")}; gaps = []
    for task in TASKS:
        for budget in ("steps", "compute"):
            B, tr = tuned(rows, tier, task, budget)
            if m not in B: no.append(f"{task} {budget}: not run"); continue
            miss = [b for b in BASELINES if b != m and b not in B]
            if miss: no.append(f"{task} {budget}: baselines not run: {', '.join(miss)}"); continue
            lrs = {l for l, _ in tr[m]}
            if len(lrs) < 2: no.append(f"{task} {budget}: one lr only")
            elif B[m]["lr"] in (min(lrs), max(lrs)): no.append(f"{task} {budget}: lr {B[m]['lr']:g} is on the edge")
            under = [b for b in BASELINES if b != m and len(tr[b]) < len(tr[m])]
            if under: no.append(f"{task} {budget}: tuned less: {', '.join(under)}")
            A = items(B[m], hidden)[f"success.{cell}"]; zs = {}
            for b in BASELINES:
                for c in BASE_CELLS:
                    if (b, c) != (m, cell): zs[(b, c)] = zgap(A, items(B[b], hidden)[f"success.{c}"])
            if min(zs.values()) >= 2: wins[budget].append(task); gaps.append(float(A.mean()) - max(B[b]["scores"][f"success.{c}"] for (b, c) in zs))
            if min(zs.values()) <= -2: no.append(f"{task} {budget}: lost to {min(zs, key=zs.get)}")
            ref = B.get(REF)
            if ref:
                if B[m]["test_weights"] > 1.01 * ref["test_weights"]: no.append(f"{task}: weights {B[m]['test_weights']} over the cap")
                if B[m]["search"] != ref["search"]: no.append(f"{task}: other search budget")
                if B[m]["cell_s"].get(cell, 0) > 1.25 * ref["cell_s"]["llll"]: no.append(f"{task}: search {B[m]['cell_s'].get(cell, 0):.1f} s against {ref['cell_s']['llll']:.1f} s")
    for b, w in wins.items():
        if len(w) < 3: no.append(f"{b}: wins {len(w)} tasks of 5, 3 needed")
    sp = []
    for task in TASKS:
        B, _ = tuned(rows, tier, task, "steps")
        if REF in B: sp.append([r["scores"]["success.llll"] for r in rows if r["tier"] == tier and r["task"] == task and r["method"] == REF and r["budget"] == "steps" and r["lr"] == B[REF]["lr"] and r["knob"] == B[REF]["knob"] and not r["tag"]])
    if not sp or min(map(len, sp)) < 3: no.append("fewer than 3 fresh worlds of lewm")
    elif gaps and min(gaps) < 2 * max(np.std(x, ddof=1) for x in sp): no.append("gap under 2 x fresh-world spread")
    return not no, no


def bottleneck(rows, faults, task, tier):
    """Returns the slot, or None, and the reasons."""
    B, _ = tuned(rows, tier, task, "steps")
    if REF not in B: return None, ["lewm not run"]
    r = B[REF]; s = r["scores"]; no = []
    if s["success.oooo"] < 0.9: no.append(f"oooo {s['success.oooo']:.2f} under .9")
    if not 0.1 <= s["success.llll"] <= 0.9: no.append(f"llll {s['success.llll']:.2f} outside .1 to .9")
    if r["top_z"] < 2: no.append(f"top slot {r['top']} z {r['top_z']:+.1f}")
    if s[f"shap.{r['top']}"] <= s["noise.flip"]: no.append(f"credit {s['shap.' + r['top']]:.2f} not above noise flip {s['noise.flip']:.2f}")
    if r["top"] != "E" and s.get("probe.r2", 1) < 0.5: no.append(f"probe R2 {s['probe.r2']:.2f} under .5")
    named = named_faults(r, faults, tier, task)
    if len(named) < 4 or not all(named.values()): no.append(f"planted faults named: {named}")
    return (r["top"] if not no else None), no


def named_faults(r, faults, tier, task):
    out = {}
    for f, slot in FAULT_SLOT.items():
        x = [q for q in faults if q["tier"] == tier and q["task"] == task and q["fault"] == f and q["strength"] == 1 and q["lr"] == r["lr"] and q["world"] == r["world"]]
        if x: q = x[-1]["scores"]; out[f] = max("EPCS", key=lambda s: q[f"shap.{s}"] - r["scores"][f"shap.{s}"]) == slot
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--tiers", nargs="+", default=["5m", "1h"]); p.add_argument("--hidden", action="store_true"); p.add_argument("--design", nargs=2, action="append", default=[]); a = p.parse_args(); rows = load(); faults = load("results_tag.jsonl"); bad = []
    for m, c in a.design + [(m, "llll") for m in CONTROLS]:
        res = [judge(rows, m, c, t, a.hidden) for t in a.tiers]; win = all(r[0] for r in res); print(f"{m} {c}: {'WIN' if win else 'no'}")
        for t, r in zip(a.tiers, res): print(f"  {t}: {'; '.join(r[1][:6]) or 'passes'}" + (f" and {len(r[1]) - 6} more" if len(r[1]) > 6 else ""))
        if win and m in CONTROLS: bad.append(m)
    for task in TASKS:
        res = [bottleneck(rows, faults, task, t) for t in a.tiers]; same = res[0][0] if res[0][0] and all(r[0] == res[0][0] for r in res) else None; print(f"bottleneck on {task}: {same or 'not called'}")
        for t, r in zip(a.tiers, res): print(f"  {t}: {r[0] or '; '.join(r[1])}")
    if bad: sys.exit(f"the test is broken: control {bad} won")
