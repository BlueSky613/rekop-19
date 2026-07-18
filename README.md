# Solution B Stable Variant (uid 152) - Final v2 - 2026-07-18

## Strategy

Shape-robust scoring. Solution B trains evenly on original batches and merged
100-hand batches, uses recency half-life 4.0, and keeps a conservative ensemble so it
stays stable across validator query shapes.

## Configuration

| Item | Value |
|---|---|
| Data | 2026-07-06 through 2026-07-18, mirrored payload view, original 1906 plus merged100 1560 for 3466 rows |
| Main ensemble | LGBM x5, HistGB x2, RF400, LogReg |
| Diversity members | Two LGBM models trained with autocorr, rand, and state axes excluded, 291 features, weight 0.7 |
| Weights | recency half-life 4.0 x merged weight 1.0 |
| Safety cap | top-K 10 percent, minimum 2, above 0.5 only |
| Defenses | per-chunk try/except, NaN sanitation, deterministic fallback |

## Verification

```text
Forward test: train through 2026-07-17, evaluate on 2026-07-18
Original34: AP=0.9834 recall=0.9189 -> comp=0.9698
Merged100:  AP=0.9966 recall=0.9667 -> comp=0.9888
Speed: 90 batches x 100 hands = 2.8 seconds
Cap: exactly 9/90 flagged, adversarial input checks passed, joblib 15.7 MB
```

## Key History

| Change | Reason |
|---|---|
| Inference speed fix | Avoids recomputing features 343 times per chunk. |
| Mixed-shape training | Original-only training was weaker on live-shape proxy data. |
| Per-model feature subsets | Adds two diversity models for humanized bot patterns. |

## Deploy

```bash
pip install -r requirements.txt
python verify.py
python neurons/miner.py --netuid 126 --wallet.name student --wallet.hotkey buzz-1 --subtensor.network finney --axon.port 8095 --neuron.uid 152 --blacklist.force_validator_permit --logging.debug
```
