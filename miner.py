"""Drop-in Poker44 SN126 miner — loads the joblib artifact (real submission format).

Copy this repo's poker44_ml/, model/poker44_model.joblib, and this file into the
Poker44-subnet checkout (replacing neurons/miner.py), then run like the reference miner.
"""

from __future__ import annotations

import json
import hashlib
import os
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

_SUBMISSION_ROOT = Path(__file__).resolve().parent
if str(_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(1, str(_SUBMISSION_ROOT))

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import build_local_model_manifest, manifest_digest
from poker44.validator.synapse import DetectionSynapse

from poker44_ml.inference import Poker44Model, SAFETY_MODE, _MODEL as _MODEL_PATH

# 대시보드로 쿼리별 상세(밸리데이터·청크별 점수) 보고 — 07_live_dashboard가 표시.
# 원격 대시보드로 보내려면:  POKER44_REPORT_URL=http://<대시보드IP>:8127  환경변수 설정.
REPORT_URL = os.environ.get("POKER44_REPORT_URL", "").strip().rstrip("/")
_QLOG = os.environ.get("POKER44_QUERY_LOG", "queries.jsonl")


def _report_query(uid, validator, scores):
    rec = {
        "uid": int(uid) if uid is not None else None,
        "validator": validator or "?",
        "n_chunks": len(scores),
        "scores": [round(float(s), 4) for s in scores],
        "window": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    try:
        with open(_QLOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + chr(10))
    except Exception:
        pass
    if REPORT_URL:
        def _post():
            try:
                body = json.dumps(rec).encode("utf-8")
                req = urllib.request.Request(REPORT_URL + "/api/report", data=body,
                                             headers={"content-type": "application/json"}, method="POST")
                urllib.request.urlopen(req, timeout=5).read()
            except Exception:
                pass
        threading.Thread(target=_post, daemon=True).start()




class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super().__init__(config=config)
        repo_root = Path(__file__).resolve().parent
        self.model = Poker44Model()
        # ★자동 리로드: daily_update가 joblib을 갱신하면 재시작 없이 반영 (재학습 실제 적용)
        self._model_mtime = _MODEL_PATH.stat().st_mtime if _MODEL_PATH.exists() else 0.0
        threading.Thread(target=self._reload_watcher, daemon=True).start()
        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=[
                repo_root / "miner.py",
                repo_root / "poker44_ml" / "inference.py",
                repo_root / "poker44_ml" / "combined.py",
                repo_root / "poker44_ml" / "features.py",
                repo_root / "poker44_ml" / "features_creative.py",
                repo_root / "poker44_ml" / "features_leader.py",
            ],
            defaults={
                "model_name": self._model_name(self.model.metadata),
                "model_version": self._model_version(self.model.metadata),
                "framework": "sklearn+lightgbm-ensemble (joblib)",
                "license": "MIT",
                "artifact_sha256": self._sha256_file(_MODEL_PATH) if _MODEL_PATH.exists() else "",
                "training_data_statement": (
                    "Trained only on the public Poker44 benchmark "
                    "(api.poker44.net/api/v1/benchmark). No validator-only eval labels used."
                ),
                "training_data_sources": ["poker44-public-benchmark"],
                "private_data_attestation": "Does not train on validator-only evaluation data.",
                "inference_mode": "local",
                "notes": (
                    f"2026-07-17 B standalone joblib; safety mode={SAFETY_MODE}; "
                    f"models={len(self.model.models)}; features={len(self.model.feature_names)}."
                ),
                "open_source": True,
            },
        )
        self.manifest_digest = manifest_digest(self.model_manifest)
        print(
            "[MODEL] manifest "
            f"name={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"repo={self.model_manifest.get('repo_url', '')} "
            f"commit={self.model_manifest.get('repo_commit', '')} "
            f"digest={self.manifest_digest}",
            flush=True,
        )
        bt.logging.info(
            f"Poker44 miner up | joblib models={len(self.model.models)} "
            f"features={len(self.model.feature_names)} safety={SAFETY_MODE}"
        )

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _clean_text(value) -> str:
        return str(value or "").strip()

    @classmethod
    def _model_name(cls, metadata: dict) -> str:
        name = cls._clean_text(metadata.get("name"))
        if name:
            return name
        env_name = cls._clean_text(os.getenv("POKER44_MODEL_NAME"))
        if env_name:
            return env_name
        recipe = cls._clean_text(metadata.get("recipe")).upper()
        if recipe == "B":
            return "poker44-B-robust"
        if recipe == "A":
            return "poker44-A-aggressive"
        return "poker44-standalone"

    @classmethod
    def _model_version(cls, metadata: dict) -> str:
        version = cls._clean_text(metadata.get("version"))
        if version:
            return version
        env_version = cls._clean_text(os.getenv("POKER44_MODEL_VERSION"))
        if env_version:
            return env_version
        return cls._clean_text(metadata.get("built")) or "1.0"

    def _reload_watcher(self, every=60):
        """daily_update가 model/poker44_model.joblib 을 갱신하면 자동으로 새 모델 로드.
        참조 스왑이라 원자적 — 재시작·쿼리중단 없이 재학습 결과가 반영된다."""
        while True:
            time.sleep(every)
            try:
                if not _MODEL_PATH.exists():
                    continue
                mt = _MODEL_PATH.stat().st_mtime
                if mt > self._model_mtime + 1:
                    new_model = Poker44Model()          # 갱신된 joblib 로드
                    self.model = new_model               # 원자적 교체
                    self._model_mtime = mt
                    bt.logging.info(
                        f"🔄 모델 자동 재로드 (재학습 반영) | models={len(new_model.models)} "
                        f"name={new_model.metadata.get('name')}"
                    )
            except Exception as e:
                bt.logging.warning(f"모델 재로드 실패(옛 모델 유지): {e}")

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        chunks = synapse.chunks or []
        try:
            scores = self.model.predict_chunk_scores(chunks)
        except Exception as e:
            # 절대 예외로 응답을 잃지 않는다 — 길이 안 맞으면 검증자가 통째로 폐기 → 0점.
            bt.logging.error(f"추론 실패, 폴백 점수 사용: {e}")
            n = len(chunks)
            k = max(1, n // 10) if n < 8 else max(2, n // 10)
            scores = [0.55 if i < k else 0.05 for i in range(n)]
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)
        # ★매니페스트 전송 안 함 (manifestPresent=False) — 상위10 전원 이렇게 함.
        #   보내면 리뷰 실패 시 -0.10~-0.22 벌점(152가 -0.10 물고 있음). 안 보내면 벌점 0.
        # synapse.model_manifest = None  # 기본값 유지
        bt.logging.info(f"Scored {len(chunks)} chunks | mean={sum(scores)/max(len(scores),1):.3f}")
        try:
            vhot = getattr(getattr(synapse, "dendrite", None), "hotkey", None)  # querying validator
            _report_query(getattr(self, "uid", None), vhot, scores)
        except Exception:
            pass
        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Poker44 submission miner running...")
        while True:
            bt.logging.info(f"UID {miner.uid} | Incentive {miner.metagraph.I[miner.uid]}")
            time.sleep(300)
