# rekop-19 (uid 232) — Poker44 (Bittensor subnet 126) miner

A bot-detection miner for Poker44. The validator sends a batch of chunks (each chunk
is one subject's hands); the miner returns one score per chunk, higher meaning more
bot-like.

Model name `poker44-wide-pct`. 808 features, all read as **within-batch percentile
ranks**, scored by a 10-member weighted-rank ensemble.

---

## 1. What makes this build different

**Every feature is a rank inside the served batch, never an absolute value.**

Live play is not the benchmark's game. Measured on captured validator payloads
against the benchmark the model trains on:

| | live | benchmark |
|---|---|---|
| starting stack | 100bb | 239bb |
| seats per hand | 6.45 | 4.70 |
| fold rate | 33% | 58% |
| raise rate | 4% | 13% |
| median bet | 1.03bb | 35.7bb |

An absolute-valued build scored a hundred live chunks into a band 0.008 wide. Its
ranking was noise, and average precision collapses to the bot prevalence when the
ranking is noise. Ratio features do not save you either — fold share is already a
ratio and still moved 0.62 to 0.33, because the whole population shifted together.

A percentile encodes only *who is more X than whom in this query*, which a
population-wide shift leaves untouched. Verified: rescaling every input by 0.42
changes the output by 2.2e-16.

**808 features, not fewer.** Feature reduction was tested and rejected. Standalone
recall at 5% FPR by basis size: 0.750 at 20 columns, 0.783 at 30, 0.783 at 50, 0.800
at 100, 0.817 at 200, 0.833 at 705. Monotone. Against the 705-column basis at
matched training windows, this 808-column union wins at every window length.

**A 21-day rolling training window.** Six holdout days, evaluated on the merged-100
chunk shape that live queries actually use:

| window | mean AP | mean recall @5% FPR |
|---|---|---|
| 14 days | 0.9814 | 0.9111 |
| **21 days** | 0.9827 | **0.9306** |
| 28 days | 0.9834 | 0.9222 |
| whole archive | 0.9831 | 0.9194 |

Read this as "three weeks or more, and 14 days is too few". The spread among 21, 28
and the whole archive is under one holdout chunk per day. The window rolls, so a
daily retrain needs no edit.

**Flags the top 10% of each query.** The reward zeroes any miner whose scores never
cross 0.5 on a real bot, so the miner must flag something. It also penalises a miner
whose flagged-human rate exceeds 10%. Flagging a tenth of the batch satisfies the
first without approaching the second.

---

## 2. Setup

```bash
git clone https://github.com/BlueSky613/rekop-19.git
cd rekop-19
```

Put the trained artifact at `model/poker44_model.joblib`. **It is not in this
repository** — a clone alone will not start.

Isolated virtualenv:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install "bittensor<11" "websockets==14.2"
```

`bittensor` 11.x removed `Synapse`; 9.x is too old for current chain metadata.
`websockets` past 14.2 raises `ConcurrencyError` and can block axon serving.

The artifact is pickled with scikit-learn 1.7.2, lightgbm 4.6.0, xgboost 3.2.0,
catboost 1.2.10, numpy 2.2.6 on python 3.10. A large version gap can stop it
unpickling, and then the miner does not start at all.

Confirm the model loads, **from the repository root**:

```bash
python -c "from poker44_ml.inference import Poker44Model; m=Poker44Model(); print(len(m.feature_names), len(m.models))"
```

Expect `808 10`. A different feature count means you are in the wrong directory or
have the wrong artifact.

---

## 3. Register and run

Register with the **same OS account** that will run the miner. A hotkey file created
by another user is unreadable and the miner crash-loops on `KeyFileError` while the
process supervisor still reports it online.

```bash
btcli subnet register --netuid 126 --wallet.name <coldkey> --wallet.hotkey <hotkey>
```

```bash
pm2 start miner.py --name p44-232 --interpreter .venv/bin/python3 -- \
  --netuid 126 --wallet.name <coldkey> --wallet.hotkey <hotkey> \
  --axon.ip 0.0.0.0 --axon.port <PORT> \
  --axon.external_ip <PUBLIC_IP> --axon.external_port <PORT> \
  --subtensor.network finney
pm2 save && pm2 startup     # without startup, a reboot loses the day
```

`--axon.external_ip` and `--axon.external_port` are **mandatory**. Without them the
axon publishes as `0.0.0.0:0`, no validator can reach you, and you score nothing
while the process looks perfectly healthy. The port must also be open inbound.

---

## 4. Verify

Expected startup lines:

```
[MODEL]   loaded joblib ... models=10 features=808
[MODEL]   manifest name=poker44-wide-pct version=... repo=... commit=... digest=...
[STARTUP] axon served on-chain uid=<UID> external=<PUBLIC_IP>:<PORT>
[HEARTBEAT] uid=<UID> block=...
```

On chain:

```bash
python -c "import bittensor as bt; mg=bt.Metagraph(netuid=126, network='finney'); print(mg.axons[<UID>].ip, mg.axons[<UID>].port)"
```

It must print your public IP and port, not `0.0.0.0 0`.

### The one number to watch: `raw_std`

Every validator query logs:

```
[FORWARD] received chunks=100
[DIAG] n=100 raw_mean=... raw_std=0.3... hard_flags=10
```

`raw_std` is the spread of the model's own scores across the batch. Above ~0.2 it is
separating chunks; near zero it is not, and a `collapse` marker is set.

**Do not judge this from the scores on the `[FORWARD]` line.** Those always look well
spread, because the rank remap re-spaces them by position even when the model gave
every chunk the same value. Only `raw_std` shows the truth.

`hard_flags` should be 10% of the chunk count.

---

## 5. Update order matters

The miner watches its model file and reloads on change. Copying a new artifact in
while old code is still running makes that code look for column names it does not
know, find nothing, and feed the model an all-zero row for every chunk — one
identical score for the whole batch. Always:

```bash
git pull                                   # code first
scp ... model/poker44_model.joblib         # then the artifact
pm2 restart p44-232                        # then restart
```

---

## 6. Retraining

```bash
python train.py --all
```

Reads the benchmark archive from `02_benchmark_data/chunks/*.json` (or
`downloads/poker44_benchmark/<date>/chunks.jsonl`), takes the newest 21 dates,
mirrors every hand through the validator's `prepare_hand_for_miner`, builds both the
native and merged-100 chunk shapes, converts to percentiles, and writes
`model/poker44_model.joblib`.

Mirroring is not optional: training on unstripped hands teaches the model to use
fields the validator removes before we see them, and serving collapses from 0.92 to
0.00. Mixed shapes are not optional either: live queries arrive as ~100-hand batches
while the benchmark ships 30-40 hand chunks, and native-only training scores 0.78 on
the live shape against 0.95 for mixed.

---

## 7. Public model manifest

With `POKER44_SEND_MODEL_MANIFEST=1` (the default) the miner attaches a transparency
manifest to each validator response. Repository identity is read from the deployed
checkout, not from configuration:

- `repo_url` — normalised from `git config --get remote.origin.url`
- `repo_commit` — the full hash from `git rev-parse HEAD`
- `implementation_files` — `miner.py`, `train.py`, and every `poker44_ml` module that
  shapes a prediction
- `artifact_sha256` — SHA-256 of the deployed joblib

Setting `POKER44_MODEL_REPO_COMMIT` makes startup fail when it does not equal the
deployed git HEAD, which stops a stale or unpushed commit being advertised to
validators. Publish the commit before serving it: the manifest names a revision that
a reviewer has to be able to resolve.

---

## 8. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| axon shows `0.0.0.0 0` on chain | missing external ip/port flags | add `--axon.external_ip` and `--axon.external_port` |
| no `[FORWARD]` lines ever | port closed inbound, or axon unreachable | open the port; re-check the on-chain address |
| `KeyFileError`, crash loop | hotkey file owned by another user | `chown` that one hotkey file to the running user |
| `ConcurrencyError` in logs | websockets newer than 14.2 | `pip install "websockets==14.2"` |
| miner will not start, unpickling error | package version gap | match the versions in section 2 |
| `[DIAG] raw_std` near 0, `collapse` set | artifact and code disagree on column names | pull code first, then copy the artifact, then restart |
| feature count is not 808 | wrong artifact, or run from outside the repo root | run from the repo root; check the artifact |
| scored 0 for a whole window | unreachable during that window | scoring runs in 24h windows, five to an epoch, and the epoch score is their mean — a lost day is a fifth of the result |

---

## 9. The three things that actually cause a zero

1. Missing or corrupt model file — the miner never starts.
2. Missing `--axon.external_ip` / `--axon.external_port` — validators cannot reach you.
3. Port not open inbound — validators cannot reach you.

None of these show as an error in the miner log; the process looks fine while scoring
nothing. Verify the on-chain axon address and watch for `[FORWARD]` lines.

A newly registered miner sits in its immunity period and is scored from the next
competition window, not immediately.

---

## License

MIT. See `LICENSE`.
