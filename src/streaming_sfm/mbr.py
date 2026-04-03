from __future__ import annotations

import concurrent.futures
import itertools
import math
from dataclasses import dataclass
from typing import Optional
from torch import Tensor

from mbrs.metrics import Metric, register
from mbrs import timer
from dataclasses import dataclass, field
from torch import Tensor
import fast_wer


@dataclass
class _FastWerConfig(Metric.Config):
    """Shared configuration for fast WER/CER metrics."""
    strip_punctuation: bool = False
    normalize_text: bool = False
    mode: str = "wer"  # "wer" | "cer"


class _MetricFastWerBase(Metric):
    """Shared implementation for WER and CER metrics backed by fast_wer."""

    HIGHER_IS_BETTER: bool = False

    Config = _FastWerConfig

    def __init__(self, cfg: _FastWerConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.trie_stats = None

    def score(self, hypothesis: str, reference: str, *_, **__) -> float:
        """Calculate the score for a single (hypothesis, reference) pair."""
        if self.cfg.mode == "cer":
            score, *_ = fast_wer.cer(
                hypothesis, reference,
                self.cfg.strip_punctuation,
                self.cfg.normalize_text,
            )
        else:
            score, *_ = fast_wer.wer(
                hypothesis, reference,
                self.cfg.strip_punctuation,
                self.cfg.normalize_text,
            )
        return score

    def pairwise_scores(
        self, hypotheses: list[str], references: list[str], *_, **__
    ) -> Tensor:
        """Calculate the H × R pairwise score matrix."""
        scorer = fast_wer.PrefixCachedScorer(
            hypotheses,
            references,
            self.cfg.strip_punctuation,
            self.cfg.normalize_text,
        )
        self.trie_stats = scorer.trie_stats()
        if self.cfg.mode == "cer":
            flat = scorer.cer_matrix()
        else:
            flat = scorer.wer_matrix()
        return Tensor(flat).reshape(scorer.shape())


@register("fastwer")
class MetricFastWer(_MetricFastWerBase):
    """Word Error Rate metric backed by fast_wer."""

    @dataclass
    class Config(_FastWerConfig):
        mode: str = field(default="wer", init=False)


@register("fastcer")
class MetricFastCer(_MetricFastWerBase):
    """Character Error Rate metric backed by fast_wer."""

    @dataclass
    class Config(_FastWerConfig):
        mode: str = field(default="cer", init=False)