# Reviewer pass 3: from the start, checking every claim about other papers against their text

Every number in `claims.py`, `README.md` and `EXPERIMENTS.md` was searched for in the saved paper text. This pass found new issues and fixed them. It can't close stage 3 of the method, because the open items in pass 1 and pass 2 need real-size runs.

| # | Issue | State |
|---|---|---|
| 1 | VisualPatchWorld was going to be cited only as a side note. Its method says it can "attribute failures to perception, dynamics, or scoring". | Fixed. It is in the closest-work and novelty tables as the closest swap design, with the difference stated: code world models, no search slot split, no credit split and no planted faults. |
| 2 | I had noted NGC and IMWM as a plain conflict on TwoRoom. NGC's appendix E discusses why a true predictor can lower success there, and NGC also holds with the true predictor. | Fixed. The text says they conflict on what the limit is, and that both can be right, which is what the 16 cells can show. |
| 3 | VIScore (2608.11174) is withdrawn on arXiv. | Fixed. Nothing cites it as evidence. |
| 4 | DWM's 30% figure is the share of zero-action frames added, and the second 30% is the success it reaches. It is easy to misread. | Fixed. The text says both in full. |
| 5 | SLIM's result is a success rate on 1200 trials, so ".003" is 3 of 1200. | Fixed. Both forms are given. |
| 6 | IMWM's oracle test covers two tasks with a single oracle adapter, and leaves Push-T out. | Stated in the closest-work table, so its claim isn't stretched. |
| 7 | The papers' numbers come from other tasks and simulators, so `claims.py` can compare only direction. | Stated in the script and in the README. |
| 8 | No claim here says the question is new, and eight related papers are unread. | Stated. |
