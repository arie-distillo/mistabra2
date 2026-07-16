"""Domain models. Plain dataclasses; the API layer wraps these in pydantic."""
from __future__ import annotations
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Optional


def _hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


@dataclass
class Document:
    doc_id: str
    name: str
    text: str
    meta: dict = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    order: int
    meta: dict = field(default_factory=dict)


@dataclass
class DataPoint:
    """An atomic, single-proposition unit. `dp_id` is content-addressed so the
    lift cache survives re-ingestion of the same text."""
    dp_id: str
    chunk_id: str
    doc_id: str
    text: str
    meta: dict = field(default_factory=dict)
    prior: Optional[float] = None   # P(D), filled by the scorer

    @staticmethod
    def make(chunk_id: str, doc_id: str, text: str, meta: dict | None = None) -> "DataPoint":
        return DataPoint(dp_id=_hash(text), chunk_id=chunk_id, doc_id=doc_id,
                         text=text.strip(), meta=meta or {})


@dataclass
class Hypothesis:
    hyp_id: str
    text: str
    context_summary: str = ""
    meta: dict = field(default_factory=dict)

    @staticmethod
    def make(text: str, context_summary: str = "", meta: dict | None = None) -> "Hypothesis":
        return Hypothesis(hyp_id=_hash(text), text=text.strip(),
                          context_summary=context_summary.strip(), meta=meta or {})


@dataclass
class LiftCell:
    """One (hypothesis, data point) score. `lift` is P(D|H)/P(D)."""
    hyp_id: str
    dp_id: str
    change: str            # ordinal label from the oracle
    lift: float
    log_lift: float
    relatedness: float
    verdict: str
    rationale: str
    model: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HypothesisResult:
    hypothesis: Hypothesis
    cells: list[LiftCell]
    score: float                     # sum of log-lift (credibility-weighted)
    worst_cell: Optional[LiftCell]   # most-contradicting data point
    best_cell: Optional[LiftCell]    # most-supporting data point
    verdict: str
    n_supporting: int
    n_contradicting: int
    n_neutral: int


@dataclass
class Matrix:
    hypotheses: list[Hypothesis]
    data_points: list[DataPoint]
    cells: dict                      # (hyp_id, dp_id) -> LiftCell
    results: list[HypothesisResult]
    diagnosticity: dict              # dp_id -> variance of log-lift across hypotheses
