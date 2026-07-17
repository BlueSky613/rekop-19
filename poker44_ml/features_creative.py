"""창의 피처 (Solution 2) — 기존 combined312에 얹는 신규 피처.

설계 원칙 (검증된 노이즈 분석 반영):
  · 범주형 축(액션종류·스트리트·플레이어)만 사용 — 노이즈 없음.
  · 베팅액·비연속시퀀스 축 회피 — 노이즈로 무너짐.
  · 봇=기계적 규칙성 → 자기유사성·저분산·고상관을 다각도로 포착.

creative_features(chunk) -> dict  (chunk = 미러링된 미너-가시 핸드 리스트)
"""
from __future__ import annotations
import math
from collections import Counter

ACTS = ("fold", "check", "call", "bet", "raise")
STREETS = ("preflop", "flop", "turn", "river")
AGG = {"bet", "raise"}


def _mean(a): return sum(a) / len(a) if a else 0.0
def _var(a):
    if len(a) < 2: return 0.0
    m = _mean(a); return sum((x - m) ** 2 for x in a) / len(a)
def _std(a): return math.sqrt(_var(a))


def _skew(a):
    if len(a) < 3: return 0.0
    m, s = _mean(a), _std(a)
    if s < 1e-9: return 0.0
    return sum(((x - m) / s) ** 3 for x in a) / len(a)


def _kurt(a):
    if len(a) < 4: return 0.0
    m, s = _mean(a), _std(a)
    if s < 1e-9: return 0.0
    return sum(((x - m) / s) ** 4 for x in a) / len(a) - 3.0


def _entropy(counts):
    tot = sum(counts)
    if tot <= 0: return 0.0
    return -sum((c / tot) * math.log(c / tot + 1e-12) for c in counts if c > 0)


def _hand_profile(hand):
    """한 핸드의 범주형 프로파일 (노이즈 없는 축만)."""
    actions = hand.get("actions") or []
    players = hand.get("players") or []
    streets = hand.get("streets") or []
    c = Counter(a.get("action_type") for a in actions)
    n = max(1, sum(c.get(k, 0) for k in ACTS))
    prof = {k: c.get(k, 0) / n for k in ACTS}
    prof["_agg"] = (c.get("bet", 0) + c.get("raise", 0)) / n
    prof["_nact"] = len(actions)
    prof["_nstreet"] = len(streets)
    prof["_nplayer"] = len(players)
    prof["_seq"] = "".join((a.get("action_type") or "?")[:1] for a in actions)
    return prof


def _cosine(v1, v2):
    keys = ACTS
    a = [v1[k] for k in keys]; b = [v2[k] for k in keys]
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-9 or nb < 1e-9: return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def creative_features(chunk):
    out = {}
    n = len(chunk)
    out["cr_n_hands"] = float(n)
    if n == 0:
        return out
    profs = [_hand_profile(h) for h in chunk]

    # ---- 1) 자기유사성 다중메트릭 (봇=서로 닮음) ----
    # (a) 액션분포 쌍별 코사인 유사도 평균 (높을수록 봇)
    sims = []
    for i in range(min(n, 40)):
        for j in range(i + 1, min(n, 40)):
            sims.append(_cosine(profs[i], profs[j]))
    out["cr_sim_action_mean"] = _mean(sims)
    out["cr_sim_action_std"] = _std(sims)      # 유사도 자체의 분산 (봇=낮음)
    # (b) 시퀀스 고유성 (낮을수록 반복=봇)
    seqs = [p["_seq"] for p in profs]
    out["cr_seq_uniq_ratio"] = len(set(seqs)) / n
    grams = Counter()
    for s in seqs:
        for k in range(len(s) - 1):
            grams[s[k:k + 2]] += 1
    tot = sum(grams.values()) or 1
    out["cr_seq_bigram_entropy"] = _entropy(list(grams.values()))
    out["cr_seq_top_bigram_share"] = max(grams.values()) / tot if grams else 0.0

    # ---- 2) 분포 고차모멘트 (평균 너머) ----
    for key, name in [("_agg", "agg"), ("call", "call"), ("fold", "fold"),
                      ("raise", "raise"), ("_nstreet", "street"), ("_nplayer", "player")]:
        vals = [p[key] for p in profs]
        out[f"cr_{name}_std"] = _std(vals)
        out[f"cr_{name}_skew"] = _skew(vals)
        out[f"cr_{name}_kurt"] = _kurt(vals)

    # ---- 3) 피처 상호작용 (top 판별축의 곱·비) ----
    sim = out["cr_sim_action_mean"]
    agg_std = out["cr_agg_std"]
    uniq = out["cr_seq_uniq_ratio"]
    out["cr_x_sim_by_aggstd"] = sim / (agg_std + 1e-6)      # 유사성↑·분산↓ = 강한 봇신호
    out["cr_x_sim_times_uniqinv"] = sim * (1.0 - uniq)       # 유사·반복 동시
    out["cr_x_uniq_by_streetstd"] = uniq / (out["cr_street_std"] + 1e-6)
    out["cr_x_aggstd_times_streetstd"] = agg_std * out["cr_street_std"]

    # ---- 4) 교차핸드 규칙성 (봇=일관) ----
    # 핸드별 공격성의 자기상관(lag1) — 봇은 패턴 반복
    agg_seq = [p["_agg"] for p in profs]
    if len(agg_seq) >= 3 and _std(agg_seq) > 1e-9:
        a, b = agg_seq[:-1], agg_seq[1:]
        ma, mb = _mean(a), _mean(b)
        cov = sum((x - ma) * (y - mb) for x, y in zip(a, b)) / len(a)
        out["cr_agg_autocorr"] = cov / (_std(a) * _std(b) + 1e-12)
    else:
        out["cr_agg_autocorr"] = 0.0
    # 전반/후반 드리프트 (봇=드리프트 없음)
    if n >= 6:
        h = n // 2
        out["cr_half_agg_drift"] = abs(_mean(agg_seq[:h]) - _mean(agg_seq[h:]))
        out["cr_half_uniq_drift"] = abs(
            len(set(seqs[:h])) / h - len(set(seqs[h:])) / (n - h))
    else:
        out["cr_half_agg_drift"] = out["cr_half_uniq_drift"] = 0.0

    return out


if __name__ == "__main__":
    import sys, json, copy
    sys.path.insert(0, "E:/BIT/127/127/10_relearn_2026-07-15/Poker44-subnet-fresh")
    from poker44.validator.payload_view import prepare_hand_for_miner
    d = json.load(open("E:/BIT/127/127/02_benchmark_data/chunks/2026-07-15.json", encoding="utf-8"))
    sc = d["chunks"][0]
    print("=== 창의 피처 봇 vs 사람 판별력 테스트 ===")
    import numpy as np
    feats, ys = [], []
    for bag, y in zip(sc["chunks"], sc["groundTruth"]):
        hands = [prepare_hand_for_miner(copy.deepcopy(h)) for h in bag]
        feats.append(creative_features(hands)); ys.append(y)
    cols = sorted(feats[0].keys())
    ys = np.array(ys)
    M = np.array([[f.get(c, 0.0) for c in cols] for f in feats])
    print(f"창의 피처 수: {len(cols)}")
    print("가장 잘 가르는 top8:")
    disc = []
    for j, c in enumerate(cols):
        b, h = M[ys == 1, j], M[ys == 0, j]
        if len(b) and len(h):
            sd = M[:, j].std() + 1e-9
            disc.append((abs(b.mean() - h.mean()) / sd, c, b.mean(), h.mean()))
    for s, c, bm, hm in sorted(disc, reverse=True)[:8]:
        print(f"  {c:28} 봇{bm:7.3f} 사람{hm:7.3f} 판별력{s:.2f}")
