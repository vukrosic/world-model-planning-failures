# Experiments

## Question

When planning with a learned world model fails, which part failed: the encoder, the predictor, the cost or the search? Does the answer change with the task, the goal's distance and the training budget, once all four parts are swapped on the same episodes?

## Closest work

| Paper | Part it blames | What it reports | Read |
|---|---|---|---|
| [IMWM, 2606.01626](https://arxiv.org/abs/2606.01626) | Search | The true environment in place of the predictor, under the same encoder, cost and search budget: Two-Room 87.7 to 85.5, OGBench-Cube 66.2 to 76.8. Failed episodes where no sample reached the goal at any logged replan: 86 of 87 and 127 of 139. Its own method reaches 99.2 and 94.7. | Formulas, tables and theorem. Push-T is left out of the oracle test because of a simulator mismatch. |
| [NGC, 2610.25013](https://arxiv.org/abs/2610.25013) | Cost | Changing only the cost to a shortest path on a graph of 5,000 training latents takes the farthest offset from 27% to 94.6% on TwoRoom and from 81% to 99.4% on Reacher. It still wins with the true environment as predictor, and four times as many samples don't close the gap for L2. | In full. Closest on the cost side, and Eq. 2 is ported. |
| [SLIM, 2610.03137](https://arxiv.org/abs/2610.03137) | Encoder | LeWM succeeds on 3 of 1200 small-object trials, and an inverse-dynamics term raises success to about .35. Weights (.09, 1, 1). The head is dropped at test time. | Formulas and tables. |
| [DWM, 2607.18715](https://arxiv.org/abs/2607.18715) | Predictor and its objective | LeWM falls from 94.0% on PushT to 32% with a world effect, and adding 30% zero-action data leaves it at 30%. | Formulas and tables. |
| [Jacobian centroids, 2609.33940](https://arxiv.org/abs/2609.33940) | Encoder and predictor together | Associated episodes reach 45.6% and 89.7% out of distribution, and dissociated ones 6.4% and 9.9%. Resampling flagged plans adds 7.6 and 13.2 points. | Formulas and tables. |
| [ACPC, 2608.12939](https://arxiv.org/abs/2608.12939) | Predicted cost | Invariance Radius and Separation Rate from clean and perturbed rollouts. It measures predicted cost, not task success. | Formulas. |
| [VisualPatchWorld, 2607.25236](https://arxiv.org/abs/2607.25236) | Perception, dynamics or scoring | Splits replan observation (tool or simulator state) from scoring model (induced, hybrid, simulator) on four tasks, to attribute failures in code world models. It splits no latent model and uses no planted faults. | Method and main table. Closest on the swap design. |
| [LeWM, 2603.19312](https://arxiv.org/abs/2603.19312) | None | The baseline. 300 samples, 30 elites, 30 search steps on PushT and 10 elsewhere, horizon 5 blocks with frame skip 5. | Method, appendix C.2 for PLDM's terms. |

Also found and not read in full: [FF-JEPA, 2606.09311](https://arxiv.org/abs/2606.09311), [Spectral-Target, 2609.04264](https://arxiv.org/abs/2609.04264), [JEPA-TTT, 2610.00722](https://arxiv.org/abs/2610.00722), [Representation geometry, 2608.16287](https://arxiv.org/abs/2608.16287), [Traj-LeWM, 2608.14125](https://arxiv.org/abs/2608.14125), [D-JEPA, 2609.24749](https://arxiv.org/abs/2609.24749), [Temporal-Distance JEPA, 2607.25337](https://arxiv.org/abs/2607.25337), [Value-guided JEPA, 2601.00844](https://arxiv.org/abs/2601.00844). [VIScore, 2608.11174](https://arxiv.org/abs/2608.11174) was read and is withdrawn on arXiv, so nothing here cites it as evidence.

## Why the papers don't answer it

Each paper changes one part, on the tasks where that part fails, and blames it. Two of them conflict on the same kind of task. On TwoRoom, IMWM finds that a true predictor doesn't help and concludes that the search is the limit, while NGC changes only the cost and goes from 27% to 94.6% at the farthest offset. Both can be right, because a change in one part can't say how much of the gap the other parts hold, and the gain from fixing one part depends on the state of the others.

**The difference from all of them, in one sentence:** all four parts are swapped on the same episodes across five tasks, the credit is split with interactions printed, and the method is first proven on planted faults with known answers, which also grades the published diagnostics.

## Boring explanations to rule out

1. Failures are just distance. Every score is split by the goal's offset (25, 50, 75 blocks of the expert's path).
2. A slot's credit is only planner noise. `noise.flip` is the share of episodes that flip when only the planner's noise changes.
3. The model never saw those states. `cover` and `auc.cover` measure how far the start is from the training data.
4. The cost slot's credit is the probe's error. With a learned predictor, the true cost reads states through a probe, so `probe.r2` is recorded and Rule 2 refuses a call under .5.
5. The result depends on the success threshold. `success2x` repeats every cell at twice the threshold.
6. The result depends on how the true cost was shaped. `--shape 0` removes the pull towards the point behind the disk.
7. The result is a bigger search and not a better one. The search slot's oracle changes samples and starts together, and `beat.py` caps a design's search time.

## Kill test: written before any comparison ran

`kill.py` applies these. Nothing below has run at a real size. That is 44 rows at `5m`, a target of about 4 hours on one thread that hasn't been timed, and it shouldn't run on a busy laptop.

```bash
python3 check.py --slow
python3 speed.py 5m --seconds 300
python3 jobs.py 5m --methods lewm --tasks reach rooms push small wind --lrs 0.0003 0.001 0.003
python3 jobs.py 5m --methods lewmidm --tasks small --lrs 0.0003 0.001 0.003
python3 jobs.py 5m --methods broken wide --tasks push --lrs 0.0003 0.001 0.003
# Then run each fault at the learning rate tune.dist picked for lewm on that task:
python3 jobs.py 5m --methods lewm --tasks push --lrs PUSH_TUNED_LR --fault blind
python3 jobs.py 5m --methods lewm --tasks push --lrs PUSH_TUNED_LR --fault noact
python3 jobs.py 5m --methods lewm --tasks push --lrs PUSH_TUNED_LR --fault flip
python3 jobs.py 5m --methods lewm --tasks push --lrs PUSH_TUNED_LR --fault tiny
# Repeat the four fault lines for reach, rooms, small and wind.
python3 kill.py --tier 5m
python3 faults.py 5m
```

| Test | Failure number | What failure means |
|---|---|---|
| K0: everything oracle succeeds | under .9 on any task | The task or the horizon is wrong, so credit has no ceiling. |
| K1: everything learned sits in the middle | between .1 and .9 on fewer than 2 tasks | There is nothing to attribute at this size. |
| K2: the top credit is above planner noise | below `noise.flip` on every task | The attribution can't see past the noise. |
| K3: planted faults are named | under 90% | The attribution can't be trusted at this size. |
| K4: the published disagreement shows on `rooms` at the far offset | neither the graph cost nor the true predictor gains | The toy lacks the failure the papers argue about, so it can't settle them. |
| K5: inverse dynamics beats LeWM on `small` | no gain | The published encoder fix isn't reproduced here. |
| K6: controls | `broken` gets credit with z of 2 or more, or `wide` doesn't show with z of 2 | The test rewards a blank model, or it can't see model quality. |

Failing K0 to K5 at `5m` kills this size as a place to study the question, and `1h` has to run next. The question itself dies if K3 fails at `1h`, or if K4 and K5 fail there too and an adapter to the real simulators isn't worth building.

Designs tried so far: 0.

## After the kill test

```bash
python3 jobs.py 5m --methods lewm pldm frozen lewmidm --tasks reach rooms push small wind --lrs 0.0003 0.001 0.003 --knobs 0.3 1 3
python3 jobs.py 5m --methods lewm pldm frozen lewmidm --tasks reach rooms push small wind --lrs 0.0003 0.001 0.003 --budget compute
python3 jobs.py 5m --methods lewm pldm frozen lewmidm --tasks reach rooms push small wind --lrs 0.001 --budget untrained
python3 jobs.py 5m --methods lewm --tasks reach rooms push small wind --lrs TUNED_LR --worlds 1 2
python3 jobs.py 1m --methods lewm pldm frozen lewmidm --tasks reach rooms push small wind --lrs 0.0003 0.001 0.003
python3 claims.py 5m
python3 audit.py 5m 1m
python3 compare.py 5m push
python3 beat.py --tiers 5m 1h
```

To find the smallest fault the test can see, rerun each fault with `--strength 0.25 0.5 0.75` one at a time. `faults.py` prints every strength beside the change in every published diagnostic.

## Debug log

All runs lasted about 7 seconds per step at the `7s` tier, one process at a time at the lowest priority, and their scores mean nothing.

| What ran | Found | Change |
|---|---|---|
| Expert on `push` | It stalled at the point it walks around, and reached about 20% of goals. | The around point moved out, and it reaches 100%. |
| Wind on `push` | A constant wind pinned the disk to a wall. | The drift now circles the centre. |
| Probe and goal policy | They trained inside `no_grad`, so the run crashed. | Moved out. |
| Planted `noact` and `blind` on the stand-in | Every sample tied, argmin took index 0, and index 0 was the planted expert plan. The search slot got credit for a fault that wasn't its own. | The population is shuffled after the plans go in, and fewer plans go in. `check.py` now tests this. |
| Diagnostics | Separation Rate read 2.0, because the diagonal held infinity. | The diagonal is excluded. |
| `speed.py` | `frozen` was measured after pretraining only, which gave it about 4 times the steps it earned. | It now measures 3 pretraining rounds in 10, as in training. |
| `claims.py` | It printed "holds" for 0.00 against 0.00. | A score at 0 or 1 prints "cannot tell". |
| Perfect stand-in, at the check's small search | On `push`, `llll` reaches .23 and `oooo` 1.00, so a perfect model with latent distance still fails, with credit .40 on the cost and .37 on the search. | Nothing to fix. It shows the slots separate without any learned model. |

