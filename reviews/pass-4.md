# Pass 4: reading the test and the rules as an ICML reviewer

| # | Issue | What it would do | Status |
|---|---|---|---|
| 1 | At full walking speed, `reach` goal distance was .39, .41, .40 and `rooms` goals were reached inside 25 steps. | The offset split and K4's "far" goals measured nothing on those tasks. | Fixed. `PACE` is .25 there, and `check.py` tests that distance grows. The label and replan count stay, and that is disclosed. |
| 2 | `blind` ignored `--strength`, because `obs` never passed it. | Every graded strength of that fault was the full fault. | Fixed and tested. |
| 3 | No no-model floor. On `wind` the disk moves by itself. | Credit could go to a model that does no better than holding still. | Added `success.idle` and `success.random`, in K1, K6 and Rule 2. |
| 4 | K4 passed if either the graph cost or the true predictor gained. | The published disagreement could "show" in the wrong direction. | K4 needs the graph cost to gain, and to gain over the true predictor, at offset 75, with the graph connected. |
| 5 | E=o swaps the predictor, probe and graph too, and a third of its Shapley weight is zero. | "Encoder" overclaims, and its credit was compared on unequal weight. | Named in the docs. Rule 2 reads `solo` before calling a slot. |
| 6 | Two slots that only work together tie in credit, so the top-z test could never call them. | A joint bottleneck would read as no finding. | A pair is called on its interaction z, when neither gains alone. |
| 7 | The state model used the pixel model's lr. | E's credit could be a poorly tuned state model. | `tune.dist.state`, `--state-lr` and a Rule 2 check. |
| 8 | `named_faults` and `faults.py` didn't filter method, knob, budget or stand. | Stand-in rows or other settings could grade the attribution. | `fault_rows` filters all of them. |
| 9 | `--shape` rows went to `results.jsonl` under the same name. | A changed cost could win tuning, and two runs shared a file. | Own name, own file, filtered from tuning. |
| 10 | K2 compared a credit with `noise.flip`, a share of episodes. | Two different units. | `noise.credit` is the credit's standard deviation. The formula is tested by simulation in `check.py`. |
| 11 | Rule 2 read only the pixel probe. | The state column could rest on a bad probe. | Both probes are read. |
| 12 | `claims.py` read ties as holds, used unpaired point gaps, and had no margin for "about equal". | Weak agreement counted. | Paired z of 2 in the published direction, and "about equal" means the gap plus 2 standard errors stays inside .1. |
| 13 | `audit.py` let NaN checks pass. | A score with no lr check could judge. | Only the hidden check may be missing. By chance a rank check passes 37.5% of the time, which is now stated. |
| 14 | No null test across worlds, and a design with one world could win. | Episode z can't see a change of data and weights. | N0, `null_floor`, `fresh_gap`, `verdict`. |
| 15 | Rule 1 had no correction on "nothing lost", no check of baselines on the grid edge. | A loss could pass in 120 comparisons, and a weak baseline could lose. | Bonferroni, and a baseline on the edge blocks a win. |
| 16 | `broken` ran only on `push`. | A leak on another task wouldn't show. | It runs on all five, and K6 reads all of them. |
| 17 | The rules were never tested. | A rule that can't fail looks like one that passes. | `test_rules.py`. |

Open, and stated in the README: nothing has run at a real size, so whether each kill test passes is unknown. Goal-policy and graph-cost cells share `llll`'s tuning.
