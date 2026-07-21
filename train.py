"""rekop-19 training recipe -- widest basis, percentile inputs, full archive.

Inherited from the measured recipe and changed in exactly three places, each backed
by a measurement rather than a hunch:

  1. Feature basis. The union of every extractor we own (808 columns) instead of
     705. Three consecutive holdout days: mean AP 0.9876 vs 0.9844, mean recall at
     5% FPR 0.9301 vs 0.9049.

  2. Training window. The old recipe hard-coded MIN_DATE = 2026-07-06, which kept 14
     of the 57 archived days. Six holdout days, evaluated on the merged-100 shape
     only, put the optimum at three weeks -- mean AP and recall at 5% FPR of
     0.9814/0.9111 at 14 days, 0.9827/0.9306 at 21, 0.9834/0.9222 at 28, and
     0.9831/0.9194 on the whole archive. Weighted the way the reward weights them
     (0.35 AP + 0.30 recall) that is 0.6168 / 0.6231 / 0.6209 / 0.6199. The gaps
     among 21, 28 and all sit inside a single holdout chunk per day, so read the
     result as "three weeks or more, and 14 days is too few" rather than as 21 being
     precisely optimal. A rolling window keeps that true without editing a date.

  3. Percentile inputs. Every column is converted to its rank within its own
     (date, shape) group before fitting, because the serving path converts every
     column to its rank within the served batch. Live play is a different game from
     the benchmark at the population level -- 100bb stacks against 239bb, 6.45 seats
     against 4.70, a 1.03bb median bet against 35.7bb -- so absolute-valued models
     score every live chunk into a 0.008-wide band and rank noise. A percentile is
     invariant to that shift by construction.

Unchanged and load-bearing:
  - payload mirroring: without it, serving collapses 0.92 -> 0.00
  - mixed-shape training: live queries arrive as ~100-hand batches while the
    benchmark ships 30-40 hand chunks; native-only training scores 0.78 on the live
    shape, mixed scores 0.95
  - weighted member-rank blending over probability blending

Usage: python train.py --all
"""
import copy
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from poker44_ml.combined import chunk_features

OUT = HERE / "model"; OUT.mkdir(exist_ok=True)

WINDOW_DAYS = 21               # rolling, so a daily retrain needs no edit; see note 2
HALFLIFE = 3.0                 # recency inside the window
MERGED_BOOST = 1.3             # live requests use the merged 100-hand shape
MAX_MERGED_PER_DAY_LABEL = 60
PCT_SUFFIX = "__pct"           # must match Poker44Model.PCT_SUFFIX


def _source_files():
    """Return one source per date, newest WINDOW_DAYS dates only."""
    bundled_dirs = [
        HERE / "02_benchmark_data" / "chunks",
        HERE.parent / "127" / "02_benchmark_data" / "chunks",
    ]
    download_dirs = [
        HERE / "downloads" / "poker44_benchmark",
        HERE.parent / "sn21" / "downloads" / "poker44_benchmark",
    ]
    by_date = {}
    for chunk_dir in bundled_dirs:
        if not chunk_dir.exists():
            continue
        for path in sorted(chunk_dir.glob("*.json")):
            by_date.setdefault(path.stem, path)
    for download_dir in download_dirs:
        if not download_dir.exists():
            continue
        for date_dir in sorted(download_dir.iterdir()):
            path = date_dir / "chunks.jsonl"
            if date_dir.is_dir() and path.exists():
                by_date.setdefault(date_dir.name, path)
    if not by_date:
        raise RuntimeError(
            "No benchmark chunks found. Expected 02_benchmark_data/chunks or "
            "sn21/downloads/poker44_benchmark beside this project."
        )
    dates = sorted(by_date)[-WINDOW_DAYS:] if WINDOW_DAYS else sorted(by_date)
    return [(date, by_date[date]) for date in dates]


def _source_records(path):
    if path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    yield from payload["chunks"]


def _mirror():
    try:
        from poker44.validator.payload_view import prepare_hand_for_miner
        return prepare_hand_for_miner
    except Exception:
        for c in [HERE.parent / "127" / "10_relearn_2026-07-15" / "Poker44-subnet-fresh",
                  HERE.parent / "127" / "01_subnet_code" / "Poker44-subnet"]:
            if (c / "poker44" / "validator" / "payload_view.py").exists():
                sys.path.insert(0, str(c))
                from poker44.validator.payload_view import prepare_hand_for_miner
                return prepare_hand_for_miner
    raise RuntimeError("prepare_hand_for_miner not found")


def load(mirror):
    """Native batches + merged-100 batches (3 same-day same-label batches concatenated)."""
    feats, ys, dates, shapes = [], [], [], []   # shape 0=native, 1=merged100
    for d, path in _source_files():
        per_label = {0: [], 1: []}
        for rec in _source_records(path):
            for bag, y in zip(rec["chunks"], rec["groundTruth"]):
                hands = [mirror(copy.deepcopy(h)) for h in bag]   # same transform as serving
                feats.append(chunk_features(hands)); ys.append(int(y))
                dates.append(d); shapes.append(0)
                per_label[int(y)].append(hands)
        for lab in (0, 1):
            same = per_label[lab]
            cnt = 0
            for i in range(len(same)):
                if cnt >= MAX_MERGED_PER_DAY_LABEL or len(same) < 3:
                    break
                merged = (same[i] + same[(i + 1) % len(same)] + same[(i + 2) % len(same)])[:100]
                if len(merged) < 80:
                    continue
                feats.append(chunk_features(merged)); ys.append(lab)
                dates.append(d); shapes.append(1); cnt += 1
    return feats, np.array(ys, np.int32), np.array(dates), np.array(shapes)


def to_percentile(X, groups):
    """Rank each column inside its group, ties averaged, mapped to [0, 1].

    Mirrors Poker44Model._batch_percentile exactly. The group stands in for the
    served batch: at fit time that is one (date, shape) bucket, at serve time it is
    the chunk list the validator sent.
    """
    out = np.empty(X.shape, dtype=np.float64)
    for g in np.unique(groups):
        m = groups == g
        sub = X[m]
        n = sub.shape[0]
        if n <= 1:
            out[m] = 0.5
            continue
        positions = np.arange(1, n + 1, dtype=np.float64)
        block = np.empty(sub.shape, dtype=np.float64)
        for col in range(sub.shape[1]):
            values = sub[:, col]
            ranks = np.empty(n, dtype=np.float64)
            ranks[np.argsort(values, kind="mergesort")] = positions
            uniq, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
            totals = np.zeros(len(uniq), dtype=np.float64)
            np.add.at(totals, inverse, ranks)
            block[:, col] = ((totals / counts)[inverse] - 1.0) / (n - 1.0)
        out[m] = block
    return out


def main(train_on_all=True):
    mirror = _mirror()
    print(f"rekop-19: widest basis, percentile inputs, {WINDOW_DAYS}-day window, "
          "weighted-rank ensemble")
    feats, y, dates, shapes = load(mirror)
    cols = sorted({k for f in feats for k in f})
    raw = np.nan_to_num(np.array([[f.get(n, 0.0) for n in cols] for f in feats], np.float64),
                        nan=0.0, posinf=0.0, neginf=0.0)
    print(f"rows={len(y)} (native {(shapes==0).sum()}, merged100 {(shapes==1).sum()}) "
          f"features={len(cols)} bot_rate={y.mean():.3f}")

    groups = np.array([f"{d}|{s}" for d, s in zip(dates, shapes)])
    X = to_percentile(raw, groups)
    print(f"percentile groups={len(set(groups.tolist()))} "
          f"(median size {int(np.median(np.unique(groups, return_counts=True)[1]))})")

    held = [] if train_on_all else sorted(set(dates.tolist()))[-1:]
    mask = np.ones(len(y), bool) if train_on_all else np.array([d not in set(held) for d in dates])
    uniq = sorted(set(dates.tolist())); dpos = {d: i for i, d in enumerate(uniq)}
    parr = np.array([dpos[d] for d in dates], float)
    sw = (np.power(0.5, (parr.max() - parr) / HALFLIFE) *
          np.where(shapes == 1, MERGED_BOOST, 1.0))[mask]
    print(f"train rows={mask.sum()} days={len(uniq)} halflife={HALFLIFE} merged_boost={MERGED_BOOST}")

    models, weights = [], []
    for seed in range(5):
        m = lgb.LGBMClassifier(n_estimators=700, learning_rate=0.03, num_leaves=63,
                               feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
                               min_child_samples=20, random_state=seed, verbose=-1)
        m.fit(X[mask], y[mask], sample_weight=sw); models.append(m); weights.append(1.0)
    for M, w in ((RandomForestClassifier(n_estimators=500, min_samples_leaf=2, random_state=0, n_jobs=-1), 1.2),
                 (ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, random_state=0, n_jobs=-1), 1.2),
                 (HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, random_state=0), 1.2)):
        M.fit(X[mask], y[mask], sample_weight=sw); models.append(M); weights.append(w)

    # Axis-ablated diversity members - counter 'humanized bots' (a second view that cannot
    # see autocorr/rand/state axes; missed bots faked exactly those. Measured +recall +AP)
    feature_idx = [None] * len(models)
    EXCL = {i for i, c in enumerate(cols) if ("autocorr" in c) or c.startswith("rand_") or c.startswith("state_")}
    KEEP = [i for i in range(len(cols)) if i not in EXCL]
    for s in (10, 11):
        m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31,
                               feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
                               min_child_samples=25, random_state=s, verbose=-1)
        m.fit(X[mask][:, KEEP], y[mask], sample_weight=sw)
        models.append(m); weights.append(0.7); feature_idx.append(KEEP)

    # The serving path derives every column from the batch, so the artifact must ask
    # for the percentile names, not the raw ones.
    art = {"models": models, "model_weights": weights,
           "feature_names": [c + PCT_SUFFIX for c in cols],
           "calibrator": None, "model_feature_idx": feature_idx,
           "metadata": {"name": "poker44-wide-pct", "version": "wide-2026-07-21",
                        "blend": "weighted_rank", "payload_mirrored": True,
                        "recency_weighted_halflife_days": HALFLIFE,
                        "merged_boost": MERGED_BOOST, "mixed_shapes": True,
                        "inputs": "within-batch percentile ranks",
                        "safety": "topk_cap", "n_models": len(models),
                        "strategy": "widest basis, whole-archive percentile rank ensemble",
                        "trained_on_all": train_on_all, "holdout_dates": held,
                        "training_dates": uniq, "n_train": int(mask.sum())}}
    path = OUT / "poker44_model.joblib"
    joblib.dump(art, path, compress=3)
    print(f"saved {path.name} ({path.stat().st_size/1e6:.1f}MB, {len(models)} models, "
          f"{len(cols)} percentile features)")


if __name__ == "__main__":
    main(train_on_all="--all" in sys.argv)
