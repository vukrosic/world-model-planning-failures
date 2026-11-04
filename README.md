# Environment for researching why world-model planning fails

> [!IMPORTANT]
> We are using this environment to write a paper for ICML 2027. It was built in [AI Research School](https://www.skool.com/become-ai-researcher-2669). The code is under the MIT licence, so you can use the code and ideas in your own work.
>
> **This is a setup, and it has no results yet.** Bug checks pass on all five tasks, including planted faults on a perfect stand-in model. No world model has trained at a real size. Start with the kill test below.

## Ask which part failed when planning with a world model fails

A robot sees an image of the scene and an image of the goal. It imagines what many action sequences would do, and it runs the one that ends closest to the goal.

The model it imagines with is a **world model**. The **encoder** turns an image into a short vector called a **latent**.

The **predictor** takes a latent and an action, and returns the next latent. Repeating it gives an imagined future.

The **cost** scores how far an imagined future is from the goal. The usual cost is the squared distance between the two latents.

The **search** picks the action sequence with the lowest cost. The usual search is the cross-entropy method: sample many sequences, keep the best, sample around them and repeat.

So a failure can come from any of the four parts, and recent papers each blame a different one:

| Paper | Part it blames | Its evidence |
|---|---|---|
| [IMWM](https://arxiv.org/abs/2606.01626) | Search | The true environment in place of the predictor moves Two-Room from 87.7 to 85.5. |
| [NGC](https://arxiv.org/abs/2610.25013) | Cost | Changing only the cost moves TwoRoom's farthest goals from 27% to 94.6%. |
| [SLIM](https://arxiv.org/abs/2610.03137) | Encoder | An inverse-dynamics term moves small-object pushing from 3 of 1200 trials to about .35. |
| [DWM](https://arxiv.org/abs/2607.18715) | Predictor | A world effect moves LeWM from 94.0% to 32%, and more data doesn't bring it back. |

> **The question: if all four parts are swapped on the same episodes, how much of the failure belongs to each, and does the answer change with the task and the goal's distance?**

---

## Swap every part on the same episodes, and prove the method on known faults

| Part | What is here |
|---|---|
| **Tasks** | Five 2D tasks from images: `reach`, `rooms` (a wall with a door), `push`, `small` (a much smaller disk) and `wind` (a world effect that moves the disk). Each is aimed at one paper's failure. |
| **16 cells** | Encoder, predictor, cost and search are each learned or oracle, so every episode runs 16 ways with the same planner noise. Two more cells add the graph cost and a search started by a goal policy. |
| **Oracles** | Encoder: the true state instead of pixels. Predictor: the true task. Cost: the true path to the goal. Search: 16 times the samples, started with expert plans. |
| **Credit** | Each part's Shapley credit per episode, which adds up exactly to the gap between `oooo` and `llll`. Each pair's interaction prints separately, because the gain from fixing one part depends on the others. |
| **Planted faults** | Known answers, graded by `--strength`: `blind` hides the object from the encoder, `noact` removes actions from the predictor, `flip` gives the cost a wrong goal, `tiny` cuts the search. `faults.py` shows whether the credit names each one. |
| **Published diagnostics** | Prediction error, the share of episodes with no sample at the goal, the cost's rank agreement with the true cost, Jacobian centroids, Invariance Radius and Separation Rate. `faults.py` shows which of them moves under which fault. |
| **Baselines** | `lewm`, `pldm`, `frozen` and `lewmidm` share an encoder and predictor at test, and differ only in their training loss. Each searches the same learning rates and the same `--knob`. |
| **Controls** | `broken` sees a blank image and must earn nothing. `wide` has twice the width and must be visible. |
| **Rules** | `kill.py` is the kill test, `claims.py` holds published comparisons by direction, `audit.py` decides which scores can judge, and `beat.py` holds the win rule and the rule for calling a part the bottleneck, both committed before any comparison ran. |

Scores are saved per episode and split by the goal's offset, with no combined number. Hidden episodes sit behind `--final`. A new world model is one row in `model.py`, and a new cost or search is one more cell in `env.py`.

---

## Run the checks, then the kill test

```bash
pip install -r requirements.txt
python3 check.py
```

`check.py` takes about 15 seconds on one thread, and `--slow` adds repeat and resume. The kill test's commands and failure numbers are in [EXPERIMENTS.md](EXPERIMENTS.md). To run the whole method with Claude Code, use the [auto-research skill](.claude/skills/auto-research/SKILL.md).

| Tier | Encoder width, predictor width | Training steps | Trajectories | Samples, search steps | Purpose |
|---|---|---|---|---|---|
| `7s` | 4, 32 | 25 | 24 | 12, 2 | Checks only, and its scores mean nothing. |
| `1m` | 12, 128 | 300 | 200 | 48, 4 | The check tier for `audit.py`. |
| `5m` | 16, 192 | 1500 | 800 | 100, 6 | The kill test. |
| `1h` | 32, 512 | 15000 | 4000 | 300, 10 (30 on push tasks) | LeWM's samples, elites and search steps, at small tasks. |

Tier names are targets, and only `7s` has been timed.

---

## Know what this test can and can't judge

**It can judge,** once the kill test passes, how much of a failure belongs to each part on these five tasks, at these sizes, against these four oracles, and whether the attribution and the published diagnostics name a fault whose answer is known.

**It can't judge** what happens on the real simulators these papers use, because no adapter is written yet. It says nothing about ViT encoders, history longer than 1, or models that act from a learned policy, such as Dreamer.

> [!WARNING]
> - **Nothing has run at a real size.** The rule scripts have only read 7-second rows, so their thresholds are unexercised.
> - **The credit is against these oracles.** The search oracle is a larger search and not the true optimum, and the encoder oracle is a learned network on the true state. Change an oracle and the credit changes.
> - **The encoder's credit is zero by construction when the predictor and the cost are both oracle,** so read the interactions as well as the totals.
> - **The true cost is a choice.** On `rooms` it is the path through the door, and on push tasks it adds a pull towards the point behind the disk. `--shape 0` removes the pull, and a credit that changes sign under it can't be claimed.
> - **Data coverage isn't a part.** The `cover` score measures it, and a row with more data is the direct test.
> - **Actions are held for 5 steps,** so the search space is smaller than LeWM's, where each block packs 5 different actions.
> - **Only SIGReg comes from authors' code.** The other baselines are ports from formulas, and PLDM's term weights weren't in the papers read, so they start at 1 and the knob scales them.
> - **Rank agreement is coarse with four baselines.**

The full list of issues is in [reviews/pass-1.md](reviews/pass-1.md), [reviews/pass-2.md](reviews/pass-2.md) and [reviews/pass-3.md](reviews/pass-3.md).

---

## Is this new?

| Work | Parts swapped | World model | Tasks | Credit split with interactions | Planted faults |
|---|---|---|---|---|---|
| [IMWM](https://arxiv.org/abs/2606.01626) | predictor | latent | 2 | no | no |
| [NGC](https://arxiv.org/abs/2610.25013) | cost, predictor | latent | 4 | no | no |
| [SLIM](https://arxiv.org/abs/2610.03137) | encoder's training | latent | new benchmark | no | no |
| [VisualPatchWorld](https://arxiv.org/abs/2607.25236) | observation, scoring | code | 4 | no | no |
| **This repo** | encoder, predictor, cost, search | latent | 5 toy tasks | yes | yes |

This table comes from literature searches and from reading these papers. It doesn't settle that the comparison is new, and [EXPERIMENTS.md](EXPERIMENTS.md) lists related papers we did not read in full. The published papers ran on real simulators and this repo has not, so a row here is not a stronger result.
