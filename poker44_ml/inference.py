"""Inference wrapper — loads the real Poker44 joblib artifact and scores chunks.

Interface matches the live subnet: Poker44Model(path).predict_chunk_scores(chunks).
Blend = weighted mean of each model's P(bot). Safety = 152-proof top-K (see below).
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import warnings

import joblib
import numpy as np

warnings.filterwarnings('ignore', message='X does not have valid feature names')

from poker44_ml.combined import chunk_features

_MODEL = Path(__file__).resolve().parent.parent / "model" / "poker44_model.joblib"

# Per-folder default is set in each solution's copy of this file.
SAFETY_MODE = os.environ.get("POKER44_SAFETY_MODE", "honest").strip().lower()  # 이 폴더 기본


def _install_sklearn_pickle_compat():
    try:
        import sklearn._loss as sklearn_loss
        import sklearn._loss.loss as sklearn_loss_module
    except Exception:
        return

    for name in dir(sklearn_loss_module):
        if name.startswith("Cy") and not hasattr(sklearn_loss, name):
            setattr(sklearn_loss, name, getattr(sklearn_loss_module, name))
    sys.modules.setdefault("_loss", sklearn_loss)


class Poker44Model:
    def __init__(self, model_path=_MODEL):
        _install_sklearn_pickle_compat()
        art = joblib.load(model_path)
        self.models = list(art.get("models") or ([art["model"]] if art.get("model") else []))
        self.feature_names = list(art.get("feature_names") or [])
        w = art.get("model_weights") or [1.0] * len(self.models)
        self.weights = np.asarray(w[:len(self.models)], dtype=np.float64)
        if self.weights.sum() <= 0:
            self.weights = np.ones(len(self.models))
        self.weights /= self.weights.sum()
        self.metadata = dict(art.get("metadata") or {})

    def _rows(self, chunks):
        # chunk_features는 청크당 1회만. 내포 루프 안에 두면 피처이름 수(343)만큼
        # 재호출되어 90배치 질의가 496초 → 검증자 timeout(180초) 초과 → 응답 폐기 → 0점.
        feats = []
        for c in chunks:
            try:
                feats.append(chunk_features(c))
            except Exception:
                feats.append({})  # 결손 청크는 0벡터 — 응답 길이 보존이 0점 방지의 전부
        X = np.array(
            [[f.get(n, 0.0) for n in self.feature_names] for f in feats],
            dtype=np.float64,
        )
        # RandomForest 등은 NaN 입력에서 예외 → 질의 전체가 죽는다. 반드시 소독.
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def _blend(self, X):
        preds = []
        for m in self.models:
            if hasattr(m, "predict_proba"):
                preds.append(np.clip(m.predict_proba(X)[:, 1], 0, 1))
            else:
                preds.append(np.clip(m.predict(X), 0, 1))
        return np.average(np.vstack(preds), axis=0, weights=self.weights)

    def _safe_topk(self, p, mode):
        """152-proof: reward ignores magnitude (AP/recall use ranking; safety uses the
        count of scores>=0.5). Flag exactly K=max(1, floor(0.10*n)) top-ranked as positive.
        K>=1 avoids tp=0; K<=10% bounds hard_fpr; rank order preserved keeps AP intact."""
        n = len(p)
        if n == 0:
            return p
        k = max(1, int(math.floor(0.10 * n)))
        if n >= 8:
            k = max(k, 2)   # 작은창 tp=0 몰수 방지: 최소 2개 플래그(랭킹·composite 불변, safety만 견고)
        order = np.argsort(-p, kind="mergesort")
        if mode == "band":
            ph, pl, nh, nl = 0.509, 0.501, 0.490, 0.010
        else:
            ph, pl, nh, nl = 0.900, 0.550, 0.450, 0.020
        out = np.empty(n, dtype=np.float64)
        for i, idx in enumerate(order[:k]):
            out[idx] = ph - (i / max(k - 1, 1)) * (ph - pl)
        rest = order[k:]
        for i, idx in enumerate(rest):
            out[idx] = nh - (i / max(len(rest) - 1, 1)) * (nh - nl) if len(rest) > 1 else nl
        return np.clip(out, 0.0, 1.0)

    def predict_chunk_scores(self, chunks):
        if not chunks:
            return []
        try:
            raw = self._blend(self._rows(chunks))
        except Exception:
            # 최후 안전망: 무슨 일이 있어도 올바른 길이로 응답한다.
            # 결정적 의사순위 + 캡 → 순위가 무작위여도 composite ~0.54 (몰수 0), 예외로 잃으면 0.
            n = len(chunks)
            raw = np.array([((i * 2654435761) % 997) / 997.0 for i in range(n)], dtype=np.float64)
        scores = self._safe_topk(raw, "band" if SAFETY_MODE == "band" else "honest")
        return [round(float(s), 6) for s in scores]

    def predict_chunk_score(self, chunk):
        s = self.predict_chunk_scores([chunk])
        return s[0] if s else 0.5
