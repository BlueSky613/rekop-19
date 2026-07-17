"""Solution B stable recipe v4 for 2026-07-17.

Key measured choices:
  - Mirror the served payload view; skipping this breaks live behavior.
  - Train on mixed shapes: original 30-40 hand batches plus merged 100-hand batches.
  - Use equal merged-batch weight so the model stays robust across query shapes.
  - Use recency half-life 4, which was best in local tests.
  - Use a conservative ensemble: LGBM x3, HistGB x2, RF, and LogReg.

Usage: python train.py --all
"""
import copy
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from poker44_ml.combined import chunk_features


def _root(p):
    for _ in range(7):
        if (p / "02_benchmark_data").exists():
            return p
        p = p.parent
    return p


ROOT = _root(HERE)
CHUNK_DIR = ROOT / "02_benchmark_data" / "chunks"
OUT = HERE / "model"; OUT.mkdir(exist_ok=True)

MIN_DATE = "2026-07-06"        # current generator era only
HALFLIFE = 4.0                 # best measured half-life
MERGED_BOOST = 1.0             # equal shape weighting
MAX_MERGED_PER_DAY_LABEL = 60


def _mirror():
    try:
        from poker44.validator.payload_view import prepare_hand_for_miner
        return prepare_hand_for_miner
    except Exception:
        for c in [ROOT / "10_relearn_2026-07-15" / "Poker44-subnet-fresh",
                  ROOT / "01_subnet_code" / "Poker44-subnet"]:
            if (c / "poker44" / "validator" / "payload_view.py").exists():
                sys.path.insert(0, str(c))
                from poker44.validator.payload_view import prepare_hand_for_miner
                return prepare_hand_for_miner
    raise RuntimeError("prepare_hand_for_miner not found")


def load(mirror):
    """Load original batches plus merged 100-hand batches from same-day labels."""
    feats, ys, dates, shapes = [], [], [], []   # shape 0=original, 1=merged100
    for p in sorted(CHUNK_DIR.glob("*.json")):
        if p.stem < MIN_DATE:
            continue
        b = json.loads(p.read_text(encoding="utf-8"))
        d = b["release"]["sourceDate"]
        per_label = {0: [], 1: []}
        for rec in b["chunks"]:
            for bag, y in zip(rec["chunks"], rec["groundTruth"]):
                hands = [mirror(copy.deepcopy(h)) for h in bag]   # match serving transform
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


def main(train_on_all=True):
    mirror = _mirror()
    print("Solution B v4: mirrored payload + balanced mixed shapes + half-life 4 + conservative ensemble")
    feats, y, dates, shapes = load(mirror)
    cols = sorted({k for f in feats for k in f})
    X = np.array([[f.get(n, 0.0) for n in cols] for f in feats], np.float64)
    print(f"rows={len(y)} (original {(shapes==0).sum()}, merged100 {(shapes==1).sum()}) features={len(cols)} bot_rate={y.mean():.3f}")

    held = [] if train_on_all else sorted(set(dates.tolist()))[-1:]
    mask = np.ones(len(y), bool) if train_on_all else np.array([d not in set(held) for d in dates])
    uniq = sorted(set(dates.tolist())); dpos = {d: i for i, d in enumerate(uniq)}
    parr = np.array([dpos[d] for d in dates], float)
    sw = (np.power(0.5, (parr.max() - parr) / HALFLIFE) *
          np.where(shapes == 1, MERGED_BOOST, 1.0))[mask]
    print(f"train rows={mask.sum()} halflife={HALFLIFE} merged_boost={MERGED_BOOST}")

    models, weights = [], []
    for seed, (nl, ff) in enumerate([(31, 0.7), (31, 0.55), (15, 0.8)]):
        m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=nl,
                               feature_fraction=ff, bagging_fraction=0.8, bagging_freq=1,
                               min_child_samples=30, random_state=seed, verbose=-1)
        m.fit(X[mask], y[mask], sample_weight=sw); models.append(m); weights.append(1.0)
    for M, w in ((HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, random_state=0), 1.0),
                 (HistGradientBoostingClassifier(max_iter=400, learning_rate=0.03, max_leaf_nodes=15, random_state=1), 1.0),
                 (RandomForestClassifier(n_estimators=400, min_samples_leaf=3, random_state=0, n_jobs=-1), 1.0),
                 (LogisticRegression(max_iter=3000, C=0.5), 0.5)):
        M.fit(X[mask], y[mask], sample_weight=sw); models.append(M); weights.append(w)

    art = {"models": models, "model_weights": weights, "feature_names": cols, "calibrator": None,
           "metadata": {"name": "poker44-B-robust", "version": "4", "blend": "mean_proba",
                        "payload_mirrored": True, "recency_weighted_halflife_days": HALFLIFE,
                        "merged_boost": MERGED_BOOST, "mixed_shapes": True,
                        "safety": "topk_cap", "n_models": len(models),
                        "strategy": "principled: shape-balanced, regularized diverse ensemble",
                        "trained_on_all": train_on_all, "holdout_dates": held}}
    path = OUT / "poker44_model.joblib"
    joblib.dump(art, path, compress=3)
    print(f"saved {path.name} ({path.stat().st_size/1e6:.1f}MB, {len(models)} models)")


if __name__ == "__main__":
    main(train_on_all="--all" in sys.argv)
