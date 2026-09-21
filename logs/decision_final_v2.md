# Final decision v2 (Monday 21 Sep 2026, 11:47)

**v2 DID NOT CHANGE v1.**

**CHOSEN CHECKPOINT: `checkpoints/iter08_detlong_s6_42000k.pt`** (unchanged from `logs/decision_final.md`).
**Fallback: `checkpoints/iter04_clean5m.pt`.**

## Presentation command
```
python evaluate.py --checkpoint checkpoints/iter08_detlong_s6_42000k.pt --episodes 5 --video-folder videos
```
(Tested in v1: episodes [1115, 775, 1495, 775, 775], mean 987, max 1495. Videos already in `videos/iter08_detlong_s6_42000k/`; no new videos were needed because the winner did not change.)

## What v2 did
- Step 1: every snapshot after the 08:07 job (50M, 51M, 52M, 53M) already had a 100-episode evaluation; none was missing.
- Step 2: top 3 of those by score: 51M (837), 53M (731), 50M (624).
- Step 3: confirmed them with 200 fresh episodes, seeds from 9000, same protocol as v1 (`logs/robustness_final_confirm_v2.csv`). The rows for iter04_clean5m and 42M were NOT re-run: evaluations are seeded, so they would reproduce the v1 rows (`logs/robustness_final_confirm.csv`) exactly. Deviation from the literal command noted here on purpose.

## Confirmation table (200 episodes, seeds from 9000)

| Checkpoint | Determ. mean / median / max | Sticky mean / median / max | Score | vs iter04 | vs 42M |
|---|---|---|---|---|---|
| **42M (chosen)** | 1175 / 1375 / 1495 | 738 / 600 / 1860 | **956.3** | +26.2% | -- |
| 41M | 1204 / 1230 / 1500 | 707 / 600 / 1745 | 955.2 | +26.0% | -0.1% |
| 40M | 972 / 755 / 1830 | 713 / 600 / 1955 | 842.4 | +11.1% | -11.9% |
| 51M | 912 / 1065 / 1205 | 769 / 600 / 2060 | 840.5 | +10.9% | -12.1% |
| iter04_clean5m (fallback) | 765 / 665 / 1300 | 751 / 600 / 1815 | 758.0 | -- | -20.7% |
| 53M | 758 / 575 / 1085 | 692 / 583 / 1835 | 724.7 | -4.4% | -24.2% |
| 50M | 581 / 600 / 725 | 645 / 605 / 1210 | 613.2 | -19.1% | -35.9% |

## Rule applied (unchanged)
- vs iter04_clean5m: needs score >= 795.9 and sticky max >= 1200. 51M passes (840.5, max 2060); 53M and 50M do not.
- vs v1's choice (42M, 956.3): a v2 winner must beat it by >= 3% (>= 985.0). None does (best: 51M at -12.1%). So 42M stays.

## Notes
- Same caveat as v1: 42M's advantage over iter04_clean5m comes from the deterministic setting (median 1375 vs 665); under sticky actions the two are a tie (738 vs 751 mean). 41M is statistically tied with 42M.
- 51M has the highest sticky mean and sticky max of all (769 / 2060) but a lower deterministic result; it is a reasonable alternative only if the competition environment is known to use sticky actions (0.25).
- Snapshots after 53M (the long run keeps training) were not considered in v2.
