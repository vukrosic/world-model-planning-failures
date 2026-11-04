# Reviewer pass 2: from the start, attacking the attribution itself

This pass found new issues. It can't close stage 3 of the method, because the open items need real-size runs.

| # | Issue | State |
|---|---|---|
| 1 | Shapley credit assumes each slot has one clear "fixed" version. The oracle search is 16 times the samples plus expert plans, so it isn't the true optimum. | Stated. The credit is against this oracle, and `success.oooo` of at least .9 is K0's check that it is close enough. |
| 2 | With the predictor and the cost both oracle, the encoder only supplies the start of a true rollout, so its credit there is zero by construction. That lowers the encoder's total credit. | Open and stated. `pair.*` prints each interaction, and a reader can see which cells carry a slot's credit. |
| 3 | The state encoder (`o` for E) is a learned network on the true state, not a perfect one. A slot called the encoder means pixels against state, not perfect against learned. | Stated in the README. |
| 4 | Dataset coverage isn't a slot, and DWM argues about it directly. | Stated. `cover` and `auc.cover` measure it, and a new row with more data is the direct test. |
| 5 | Episodes are shared across cells, but the planner noise is too, so a cell can look better by a lucky draw on every episode. | Fixed with a score. `noise.flip` reruns `llll` with different noise, and Rule 2 requires credit above it. |
| 6 | With four baselines, a rank agreement in `audit.py` has only 24 possible orders, so its cut-offs are coarse. | Open and stated. |
| 7 | PLDM's seven term weights aren't in the papers read, so they start at 1 and the knob scales them all. This may under-tune PLDM. | Open and stated. K-tests don't depend on PLDM, but a comparison with it does. |
| 8 | Only SIGReg comes from authors' code. The other baselines are ports from formulas. | Stated in the README. |
| 9 | A win could come from the graph cost's extra time, which is a larger search in disguise. | Fixed. `beat.py` Rule 1 caps a design's search at 1.25 times lewm's seconds. |
| 10 | The rule scripts have read only 7-second rows, so their thresholds are unexercised. | Open, and stated at the top of the README. |
