# Reviewer pass 1: is the comparison fair and is the test strong enough?

Read as a reviewer who wants to reject the paper. This pass found issues and fixed the ones code can fix.

| # | Issue | State |
|---|---|---|
| 1 | `frozen` was timed after pretraining only, which gave it about 4 times its fair steps at equal compute. | Fixed. `speed.py` times 3 pretraining rounds in 10. The ratio comes from the pixel model only, which is stated here. |
| 2 | LeWM uses 30 search steps on PushT and 10 elsewhere. The `1h` tier used 10 everywhere. | Fixed. `iters_push` is 30 on `push`, `small` and `wind` at `1h`. |
| 3 | The faults were all or nothing, so a passing K3 says little about small faults. | Fixed. `--strength` grades every fault, and `faults.py` prints them all. |
| 4 | Nothing compared the attribution with the published diagnostics. | Fixed. `faults.py` prints each diagnostic's change under each fault. Which diagnostic detects which fault is a question this test can answer, and no diagnostic is chosen in advance. |
| 5 | A weak probe makes the true cost read wrong states, which understates the cost slot. | Fixed. `probe.r2` is recorded, and Rule 2 refuses to call a slot under R2 of .5, unless it calls the encoder. |
| 6 | The true cost on push tasks adds a pull towards the point behind the disk. A reviewer can say this choice decides the cost slot's credit. | Open with a test. `--shape 0` removes the pull. A credit that changes sign under it can't be claimed. |
| 7 | `claims.py` printed "holds" on 0.00 against 0.00, and `rooms` "barely helps" passed when the model couldn't plan. | Fixed. Scores at 0 or 1 print "cannot tell". |
| 8 | Toy tasks are 2D with a small CNN and history 1, where the papers use ViT encoders, MuJoCo and history 3. | Open and stated in the README. Rule 2 says a call at these tiers earns a run on the real simulators, and no claim about them. |
| 9 | Actions are held for 5 steps, where LeWM packs 5 different actions per block, so the search space here is smaller. | Open and stated. |
| 10 | No RSSM or Dreamer baseline. | Open and stated. The question concerns planning from latents with a search, and those methods also act from a learned policy. |
