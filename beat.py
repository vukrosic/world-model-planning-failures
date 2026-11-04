"""Two win rules, written before any new design ran. python3 beat.py [--tiers 5m 1h] [--hidden] [--design method cell]

Rule 1: does a design beat every published world model, cost and search? A design is a method and a cell among llll, llnl, lllp or a new one. It wins only if ALL of these hold at both tiers:
1. Tuned as hard as the baselines: its lr and knob are the pair tune.dist picks, neither it nor any baseline sits on the edge of what it tried, and no baseline tried fewer pairs on any task. tune.dist reads llll, so a graph-cost or policy cell is picked by the same rule for everyone, and NGC's own settings are fixed at the paper's.
2. Equal compute and equal steps: at both budgets, on at least 3 of the 5 tasks its success beats every baseline method at every baseline cell by at least 2 standard errors of the episode-by-episode gap.
3. Nothing lost: on no task and no budget does a baseline beat it, by a z corrected for every comparison made (Bonferroni at 5% one-sided, about 3.3 for 120 comparisons).
4. Beyond training noise: on the tasks it wins, its gap to the best baseline is at least 2 standard deviations of lewm's success across fresh worlds, at least 3 worlds exist, and over at least 2 fresh worlds its mean beats that baseline's mean by the null floor below.
5. Same weights and search: test-time weights at most lewm's x 1.01, the same samples, steps and elites, and the search itself at most 1.25x lewm's llll seconds, because a cost that takes far longer is a larger search.
Fairness sentence: every baseline has the same encoder and predictor at test, because they differ only in their training loss and the training-only heads are dropped; equal compute gives each method the training steps that take lewm's time on one thread.

Rule 2: can a slot be called the bottleneck on a task? Only if ALL of these hold at the tier, and on the same slot at two tiers:
1. oooo succeeds on at least .9, and llll is between .1 and .9 and above holding still and above random plans by z 2, so there is something to explain and the task is solvable.
2. The top slot's credit is more than 2 x noise.credit, the credit's standard deviation from planner noise alone.
3. All four planted faults at full strength are named correctly at this tier, so the attribution has been shown to work at this size. A fault counts as named if its slot's credit rises most against the same row with no fault.
4. Both probes read their latents with R2 of at least .5, except the pixel probe when the slot called is E. A weak probe understates the cost slot, because the true cost then gets wrong states.
5. The state model isn't under-tuned at this lr: its tune.dist.state is within 5% of its best across the lrs tried. If E's credit depends on a poorly tuned state model, rerun with --state-lr.
6. At least 3 worlds exist.
Then the call. A single slot is called when its credit beats the second by z >= 2 episode by episode, making it oracle alone gains by z >= 2, it is on top on all worlds, and its gap to the second is above the largest gap between two worlds in its credit. Otherwise a pair is called when its interaction has z >= 2, neither slot gains alone by z 2, and the top slot on every world is in the pair, because two slots that only work together tie in credit by construction. Otherwise no call.
E means what the model reads, with the dynamics trained on it. See attrib.py.
A win or a bottleneck at these tiers earns a run on the real simulators, and no claim about them holds until that run exists. Report the number of designs tried with any win. The controls must never win: the script exits with an error if one does.

The null floor is the largest gap in success between two runs of lewm at one setting on different worlds. A world changes data, weights and episodes, so that gap contains everything an episode-by-episode z can't see.
"""
import argparse, json, os, statistics, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
from model import BASELINES, CONTROLS, REF
from world import TASKS, SHAPE
BASE_CELLS = ("llll", "llnl", "lllp"); FAULT_SLOT = {"blind": "E", "noact": "P", "flip": "C", "tiny": "S"}; low = lambda k: k.startswith("dist"); DEFAULT_SHAPE = SHAPE; LL = "success.llll"


def load(name="results.jsonl"):
    p = os.path.join(HERE, name); return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def plain(r, tier, task, budget, fault=""):
    """A row made by the normal setup at this tier, task and budget."""
    return r["tier"] == tier and r["task"] == task and r["budget"] == budget and r["fault"] == fault and not r["tag"] and not r.get("stand") and r.get("shape", DEFAULT_SHAPE) == DEFAULT_SHAPE


def tuned(rows, tier, task, budget, world=0, fault=""):
    """The row tune.dist picks per method, and every pair tried. tune.dist comes from tuning episodes, which are never reported. --state-lr rows share pixel cells, so between those the better state model is picked."""
    best, tried = {}, {}; key = lambda r: (r["scores"]["tune.dist"], r["scores"].get("tune.dist.state", 0))
    for r in rows:
        if plain(r, tier, task, budget, fault) and r["world"] == world:
            tried.setdefault(r["method"], set()).add((r["lr"], r["knob"]))
            if r["method"] not in best or key(r) < key(best[r["method"]]): best[r["method"]] = r
    return best, tried


def items(r, hidden=False):
    n = f"{r['tier']}_{r['task']}_{r['method']}_{r['budget']}_lr{r['lr']:g}_k{r['knob']:g}_w{r['world']}" + (f"_{r['fault']}" if r["fault"] else "") + (f"_s{r['strength']:g}" if r["strength"] != 1 else "") + (f"_shape{r['shape']:g}" if r.get("shape", DEFAULT_SHAPE) != DEFAULT_SHAPE else "") + (f"_slr{r['state_lr']:g}" if r.get("state_lr") else "") + (f"_{r['tag']}" if r["tag"] else "") + ".npz"; return np.load(os.path.join(HERE, "runs", ("final_" if hidden else "") + n))


def zgap(a, b):
    d = a - b; return float(d.mean() / max(d.std() / len(d) ** 0.5, 1e-9))


def at(rows, tier, task, budget, m, r0):
    """Every world of a method at the setting in row r0."""
    return {r["world"]: r for r in rows if plain(r, tier, task, budget) and r["method"] == m and r["lr"] == r0["lr"] and r["knob"] == r0["knob"] and r.get("state_lr", 0) == r0.get("state_lr", 0)}


def null_floor(rows, tier, task, budget, m=REF, k=LL):
    """(largest gap in score between two worlds of one method and setting, worlds)."""
    B, _ = tuned(rows, tier, task, budget)
    if m not in B: return 0.0, 0
    v = [r["scores"][k] for r in at(rows, tier, task, budget, m, B[m]).values()]; return (float(max(v) - min(v)) if v else 0.0), len(v)


def fresh_gap(rows, tier, task, budget, rx, ry, kx=LL, ky=LL):
    """Mean of kx minus mean of ky over the fresh worlds both have, at the settings world 0 picked. Returns (gap, worlds)."""
    A, C = at(rows, tier, task, budget, rx["method"], rx), at(rows, tier, task, budget, ry["method"], ry); ws = [w for w in A if w in C and w != 0]
    if not ws: return float("nan"), 0
    return float(np.mean([A[w]["scores"][kx] - C[w]["scores"][ky] for w in ws])), len(ws)


def verdict(rows, tier, task, budget, rx, ry, kx=LL, ky=LL):
    """Does kx of row rx beat ky of row ry beyond training noise? Returns ("passes" | "FAILS" | "not judged", text)."""
    d = items(rx)[kx] - items(ry)[ky]; g = float(d.mean()); zv = float(g / max(d.std() / len(d) ** 0.5, 1e-9)); f, n = fresh_gap(rows, tier, task, budget, rx, ry, kx, ky); floor, nw = null_floor(rows, tier, task, budget)
    text = f"{rx['method']} {kx.split('.', 1)[1]} over {ry['method']} {ky.split('.', 1)[1]} on {task} {g:+.3f}, z {zv:+.1f}, fresh worlds {f:+.3f} ({n}), null floor {floor:.3f} from {nw} worlds"
    if zv < 2: return "FAILS", text
    if n < 2 or nw < 3: return "not judged", text + ": needs at least 2 fresh worlds of each and 3 of lewm"
    return ("passes" if f > floor else "FAILS"), text


def judge(rows, m, cell, tier, hidden=False):
    no = []; wins = {b: [] for b in ("steps", "compute")}; gaps = []; n = len(TASKS) * 2 * (len(BASELINES) * len(BASE_CELLS)); lose = statistics.NormalDist().inv_cdf(1 - 0.05 / n)
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
            edge = [b for b in BASELINES if b != m and (lambda ls, ks: (len(ls) > 1 and B[b]["lr"] in (min(ls), max(ls))) or (len(ks) > 1 and B[b]["knob"] in (min(ks), max(ks))))({l for l, _ in tr[b]}, {k for _, k in tr[b]})]
            if edge: no.append(f"{task} {budget}: baselines on the edge of their grid: {', '.join(edge)}")
            A = items(B[m], hidden)[f"success.{cell}"]; zs = {}
            for b in BASELINES:
                for c in BASE_CELLS:
                    if (b, c) != (m, cell): zs[(b, c)] = zgap(A, items(B[b], hidden)[f"success.{c}"])
            if min(zs.values()) >= 2:
                wins[budget].append(task); top = max(zs, key=lambda q: B[q[0]]["scores"][f"success.{q[1]}"]); gaps.append(float(A.mean()) - B[top[0]]["scores"][f"success.{top[1]}"])
                f, nf = fresh_gap(rows, tier, task, budget, B[m], B[top[0]], f"success.{cell}", f"success.{top[1]}"); floor = null_floor(rows, tier, task, budget)[0]
                if nf < 2: no.append(f"{task} {budget}: {nf} fresh worlds of {m} and {top[0]}, 2 needed")
                elif f <= floor: no.append(f"{task} {budget}: fresh-world gap {f:+.3f} under the null floor {floor:.3f}")
            if min(zs.values()) <= -lose: no.append(f"{task} {budget}: lost to {min(zs, key=zs.get)} past the corrected z {lose:.1f}")
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
        if REF in B: sp.append([r["scores"][LL] for r in at(rows, tier, task, "steps", REF, B[REF]).values()])
    if not sp or min(map(len, sp)) < 3: no.append("fewer than 3 fresh worlds of lewm")
    elif gaps and min(gaps) < 2 * max(np.std(x, ddof=1) for x in sp): no.append("gap under 2 x fresh-world spread")
    return not no, no


def fault_rows(faults, r, tier, task):
    """Rows with a planted fault made at exactly this healthy row's setting."""
    return [q for q in faults if q["tier"] == tier and q["task"] == task and q.get("fault") and not q.get("stand") and not q["tag"] and q["method"] == r["method"] and q["budget"] == r["budget"] and q["lr"] == r["lr"] and q["knob"] == r["knob"] and q["world"] == r["world"] and q.get("shape", DEFAULT_SHAPE) == r.get("shape", DEFAULT_SHAPE) and q.get("state_lr", 0) == r.get("state_lr", 0)]


def named_faults(r, faults, tier, task):
    out = {}; here = fault_rows(faults, r, tier, task)
    for f, slot in FAULT_SLOT.items():
        x = [q for q in here if q["fault"] == f and q["strength"] == 1]
        if x: q = x[-1]["scores"]; out[f] = max("EPCS", key=lambda s: q[f"shap.{s}"] - r["scores"][f"shap.{s}"]) == slot
    return out


def bottleneck(rows, faults, task, tier):
    """Returns the slot or pair, or None, and the reasons."""
    B, _ = tuned(rows, tier, task, "steps")
    if REF not in B: return None, ["lewm not run"]
    r = B[REF]; s = r["scores"]; no = []; top = r["top"]; I = items(r); z = lambda x: float(x.mean() / max(x.std() / len(x) ** 0.5, 1e-9))
    if s["success.oooo"] < 0.9: no.append(f"oooo {s['success.oooo']:.2f} under .9")
    if not 0.1 <= s[LL] <= 0.9: no.append(f"llll {s[LL]:.2f} outside .1 to .9")
    for f in ("idle", "random"):
        if zgap(I[LL], I[f"success.{f}"]) < 2: no.append(f"llll isn't above {f} plans by z 2: {s[LL]:.2f} against {s[f'success.{f}']:.2f}")
    if s[f"shap.{top}"] <= 2 * s["noise.credit"]: no.append(f"credit {s['shap.' + top]:.2f} not above 2 x planner noise {s['noise.credit']:.3f}")
    if top != "E" and s.get("probe.r2", 1) < 0.5: no.append(f"probe R2 {s['probe.r2']:.2f} under .5")
    if s.get("probe.r2.state", 1) < 0.5: no.append(f"state probe R2 {s['probe.r2.state']:.2f} under .5")
    named = named_faults(r, faults, tier, task)
    if len(named) < 4 or not all(named.values()): no.append(f"planted faults named: {named}")
    st = [q["scores"].get("tune.dist.state", float("inf")) for q in rows if plain(q, tier, task, "steps") and q["method"] == REF and q["world"] == 0]
    if s.get("tune.dist.state", 0) > 1.05 * min(st): no.append(f"state model under-tuned at this lr: {s['tune.dist.state']:.3f} against {min(st):.3f}, so rerun with --state-lr")
    W = at(rows, tier, task, "steps", REF, r)
    if len(W) < 3: no.append(f"{len(W)} worlds, 3 needed")
    if no: return None, no
    tops = [q["top"] for q in W.values()]; credit = [q["scores"][f"shap.{top}"] for q in W.values()]; second = max(s[f"shap.{x}"] for x in "EPCS" if x != top)
    alone = r["top_z"] >= 2 and z(I[f"solo.{top}"]) >= 2 and all(t == top for t in tops) and s[f"shap.{top}"] - second > max(credit) - min(credit)
    if alone: return top, []
    pairs = {"".join(sorted(k[5:])): I[k] for k in I.files if k.startswith("pair.") and top in k[5:]}; p = max(pairs, key=lambda k: pairs[k].mean()); x = pairs[p]
    if z(x) >= 2 and all(t in p for t in tops) and all(z(I[f"solo.{c}"]) < 2 for c in p): return "+".join(p), []
    return None, [f"{top}: top z {r['top_z']:+.1f}, alone gains {I[f'solo.{top}'].mean():+.2f}, tops across worlds {tops}, gap to the second {s[f'shap.{top}'] - second:.3f} against the credit's spread {max(credit) - min(credit):.3f}, and pair {p} has z {z(x):+.1f}"]


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
