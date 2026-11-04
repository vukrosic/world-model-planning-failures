"""Run a list of settings one at a time, skipping what is done. python3 jobs.py TIER --methods lewm pldm --tasks push rooms --lrs 0.0003 0.001 0.003 [--knobs 1] [--budget steps] [--worlds 0] [--fault blind] [--strength 1] [--max-seconds 500] [--max-load 6]

--max-seconds stops the list at a clean point, and the same command continues it. Each job runs on one thread at the lowest priority in its own process, so a crash loses one job. No job starts while the machine is busy.
"""
import argparse, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); p = argparse.ArgumentParser(); p.add_argument("tier"); p.add_argument("--methods", nargs="+", required=True); p.add_argument("--tasks", nargs="+", required=True); p.add_argument("--lrs", nargs="+", type=float, required=True); p.add_argument("--knobs", nargs="+", type=float, default=[1.0]); p.add_argument("--budget", default="steps"); p.add_argument("--worlds", nargs="+", type=int, default=[0]); p.add_argument("--fault"); p.add_argument("--strength", type=float, default=1.0); p.add_argument("--max-seconds", type=float, default=0); p.add_argument("--max-load", type=float, default=6); a = p.parse_args(); t0 = time.time(); left = 0
path = os.path.join(HERE, "results_tag.jsonl" if a.fault else "results.jsonl"); done = {(r["method"], r["task"], r["lr"], r["knob"], r["world"], r["budget"], r["tier"], r["fault"], r["strength"]) for r in map(json.loads, open(path))} if os.path.exists(path) else set()
for w in a.worlds:
    for lr, knob in [(l, k) for l in a.lrs for k in a.knobs]:
        for task in a.tasks:
            for m in a.methods:
                if (m, task, lr, knob, w, a.budget, a.tier, a.fault or "", a.strength) in done: continue
                if os.getloadavg()[0] > a.max_load: sys.exit(f"load {os.getloadavg()[0]:.1f} is above {a.max_load:g}: stopping, and the same command continues later")
                lim = a.max_seconds - (time.time() - t0) if a.max_seconds else 0
                if a.max_seconds and lim <= 5: left += 1; continue
                subprocess.run([sys.executable, os.path.join(HERE, "env.py"), "--method", m, "--task", task, "--tier", a.tier, "--lr", str(lr), "--knob", str(knob), "--budget", a.budget, "--world", str(w)] + (["--fault", a.fault, "--strength", str(a.strength)] if a.fault else []) + (["--max-seconds", str(lim)] if lim else []))
print(f"jobs not started: {left}")
