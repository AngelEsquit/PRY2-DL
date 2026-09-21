# Final decision (v1, Monday 21 Sep 2026, ~09:20)

**CHOSEN CHECKPOINT: `checkpoints/iter08_detlong_s6_42000k.pt`** (long run `iter08_detlong_s6`, 42M steps; Dueling + Double DQN, trained WITHOUT sticky actions).

**Fallback: `checkpoints/iter04_clean5m.pt`** (5M steps, same config).

## Presentation command
```
python evaluate.py --checkpoint checkpoints/iter08_detlong_s6_42000k.pt --episodes 5 --video-folder videos
```
Tested at 09:2x: episodes `[1115, 775, 1495, 775, 775]`, mean 987, max 1495 (deterministic env, seeds 123..127, same as evaluate.py default).
Videos for this agent are already in `videos/iter08_detlong_s6_42000k/` (5 files).

## Why (200 fresh episodes, seeds from 9000, both settings)

| Checkpoint | Deterministic mean / median / max | Sticky (0.25) mean / median / max | Score |
|---|---|---|---|
| iter04_clean5m (fallback) | 765 / 665 / 1300 | 751 / 600 / 1815 | 758 |
| long run 40M | 972 / 755 / 1830 | 713 / 600 / 1955 | 842.5 |
| long run 41M | 1204 / 1230 / 1500 | 707 / 600 / 1745 | 955.5 |
| **long run 42M** | **1175 / 1375 / 1495** | **738 / 600 / 1860** | **956.5** |

Score = mean of (sticky mean, deterministic mean). Rule: switch away from iter04_clean5m only if a challenger's confirm score is >= 5% higher (>= 795.9) AND its sticky max >= 1200. All three challengers pass; 42M has the highest confirmed score (41M is essentially tied: 955.5).

## Caveats (read before the presentation)
- **The whole gain comes from the deterministic setting.** Under sticky actions (0.25) the 42M snapshot is a tie with iter04_clean5m (738 vs 751 mean; max 1860 vs 1815). If the instructor's environment uses sticky actions, expect roughly the same as the fallback; if it is deterministic, expect clearly better (median 1375 vs 665).
- The deterministic setting produces only ~4 distinct trajectories per 200 episodes, so its confirmation is less informative than the episode count suggests. Its 42M result (1175) is consistent with the earlier 100-episode evaluation (1203), which is reassuring, but still few independent samples.
- 40M, 41M and 42M are consecutive snapshots from one stretch of one run. Snapshots evaluated from 43M to 50M dropped back to ~585-750 score (100-episode evaluations), so 42M looks like a good moment of a fluctuating policy, not a permanently higher level. It is still the best measured.
- Snapshots that appeared after 08:07 are NOT considered in this v1. The 11:07 job (v2) may add them; v1 stays valid as is.
- Kaggle runs: iter07 (seed 5) evaluated, not competitive. iter09 (buffer 300k): output lost, only its training CSV exists; no checkpoints.
- `.pt` files are gitignored; if the agent must be in the repo, only this chosen file would need to be un-ignored (not done, needs the user).
