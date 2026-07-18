"""Combined feature extractor: leader-reproduction (293) + our honest signals (19).

Reverse-engineering payoff: we start from the CURRENT #1's exact public feature pipeline
(features_leader.py, from tao-miner/hot4-poker-3, MIT) and add the transferable honest
signals (randomization / pot-odds / state-dependence / self-similarity) that no leader uses.
On identical proxy data + temporal holdout this beats the leader reproduction.
"""

from __future__ import annotations

from poker44_ml.features_leader import chunk_features as _leader_features
from poker44_ml.features import chunk_features as _our_features
from poker44_ml.features_creative import creative_features as _creative_features

_HONEST_PREFIXES = ("rand_", "potodds_", "state_", "grid_", "simil_")


def chunk_features(chunk):
    f = dict(_leader_features(chunk))                 # 293 leader features
    ours = _our_features(chunk)
    for k, v in ours.items():                         # + only the honest signals
        if k.startswith(_HONEST_PREFIXES):
            f[k] = v
    f.update(_creative_features(chunk))               # + creative features (cr_*): interactions, higher moments, multi-similarity (validated +0.0198)
    return f
