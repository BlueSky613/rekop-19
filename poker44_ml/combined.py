"""Widest feature basis we own (rekop-19).

  leader-666  : the established leader-lineage extractor (features_leader_full)
  + ordering  : overdispersion / drift / momentum / pot-grid weapons (od_ dr_ mo_ pr_)
  + ours      : action-schema, randomness, pot-odds, state and similarity signals
  + creative  : aggression shape statistics (cr_)
  + ngram     : hand action-sequence n-grams

Union comes to 808 columns. The size was measured, not guessed: standalone recall at
5% FPR rises monotonically with basis size (0.750 at 20 columns through 0.833 at
705), and against the deployed 705 basis this union scores 0.9876 mean AP and 0.9301
mean recall over three consecutive holdout days versus 0.9844 and 0.9049.

The 373 n-gram columns overlap heavily with the leader extractor, so the union adds
103 genuinely new columns: 31 creative and 72 n-gram. Every column is read as a
within-batch percentile by the serving path, which is what makes a wide basis safe
here -- a column that drifts at the population level contributes its rank, not its
level.
"""
from __future__ import annotations

from poker44_ml.features_leader_full import chunk_features as _leader_full
from poker44_ml.features_ordering import ordering_features as _ordering
from poker44_ml.features import chunk_features as _ours
from poker44_ml.features_creative import creative_features as _creative
from poker44_ml.features_ngram import ngram_features as _ngram
from poker44_ml.features_selfconsistency import selfconsistency_features as _selfcons
from poker44_ml.features_secondorder import second_order_features as _secondorder
from poker44_ml.features_compress import compress_features as _compress


def chunk_features(chunk):
    f = dict(_leader_full(chunk))     # 666 leader-lineage columns
    f.update(_ordering(chunk))        # + 20 ordering weapons
    f.update(_ours(chunk))            # + 91 of our own signals
    f.update(_creative(chunk))        # + 31 aggression shape statistics
    f.update(_ngram(chunk))           # + hand action-sequence n-grams
    f.update(_selfcons(chunk))        # + self-consistency measures
    f.update(_secondorder(chunk))     # + per-hand statistic distributions
    f.update(_compress(chunk))        # + compression / repetition measures
    return f
