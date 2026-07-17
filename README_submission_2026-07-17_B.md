# Solution B Stable Variant (uid 152) - Final v4 - 2026-07-17

## Strategy

Robust scoring across query shapes. Current live validators usually send about 90 batches
of 100 hands, but older validator versions can send different shapes. Solution B trains
evenly on original batches and merged 100-hand batches, then uses half-life 4 and a
regularized conservative ensemble.

## Main Fixes

| Fix | Evidence |
|---|---|
| Inference speed fix: `chunk_features` runs once per chunk, not once per feature name. | 90 batches went from 496 seconds to 2.7 seconds, avoiding validator timeout. |
| Mixed-shape training: original 30-40 hand batches plus merged 100-hand batches. | Original-only training scored about 0.78 on live-shape proxy data; mixed-shape training scored about 0.95. |
| 343 features including 31 creative features. | `cr_agg_autocorr` was a strong 2026-07-17 discriminator. |

## Verification

```text
Forward test: train through 2026-07-16, evaluate on 2026-07-17
Original34 comp=0.9851
Merged100 comp=0.9591
Speed: 90 batches x 100 hands = 2.7 seconds
Safety: K=9 cap ok, empty input ok, one chunk ok, score range ok, in-sample replay 1.0000
```

Forward testing is the useful skill estimate. In-sample 1.0 is only a wiring check.

## Configuration

| Item | Value |
|---|---|
| Data | 2026-07-06 through 2026-07-17 mirrored with `prepare_hand_for_miner`; original 1758 plus merged100 1440 |
| Ensemble | LGBM x3, HistGB x2, RF400, LogReg |
| Weights | recency half-life 4.0 x merged-batch weight 1.0 |
| Safety cap | top-K = max(2, floor(0.1n)) |
| Manifest | Sent by `miner.py` for validator review |

## Deploy

```bash
python train.py --all
python verify.py

POKER44_SAFETY_MODE=honest python miner.py --axon.port <PORT> ...
```

The miner watches `model/poker44_model.joblib` and reloads it every 60 seconds when the
file changes, so a joblib replacement does not require a restart.

## A vs B

| | A (uid232) | B (this folder, uid152) |
|---|---|---|
| Half-life | 3.0, focused on the latest day | 4.0, best measured locally |
| Merged weight | 1.3, live-shape priority | 1.0, balanced |
| Ensemble | 9 larger models | 7 conservative models |
| 2026-07-17 forward test | 0.9704 / 0.9589 | 0.9851 / 0.9591 |
| Betting style | Live-shape fit | Shape-robust |
