"""Capability (b): residual detection — the hypothesis nobody wrote down.

Finds coherent clusters of data points that NO current hypothesis explains, and
abduces the missing hypothesis from each cluster. This is the differentiator:
retrieval ranks by similarity to what you asked, so it structurally cannot surface
evidence whose significance is that it relates to a hypothesis you never entered.

Three steps:
  1. residual scoring   — D is unexplained when max_H |log lift(D,H)| ~= 0.
                          Pure arithmetic over the cached grid; no LLM calls.
  2. coherence clustering — group residuals by similarity TO EACH OTHER.
  3. abduction          — one LLM call per cluster: "what hypothesis would make all
                          of these expected?", then score it back against the corpus.

CRITICAL DESIGN CONSTRAINT: cluster by data-point-to-data-point coherence, NEVER by
relevance to the existing hypotheses. The evidence for a missing hypothesis is by
definition not about the hypotheses you have; filtering on relevance destroys exactly
the signal being hunted. (An early experiment failed precisely this way.)

USING THE ABSOLUTE VALUE of log lift is deliberate: a data point that strongly
CONTRADICTS a hypothesis is explained by it (it bears on it), so it is not a residual.
Only points with no logical bearing on anything are residuals.

SCALE IS THE OPEN RISK. At hundreds of data points a coherent cluster stands out. At
tens of thousands, ambient text may produce many equally-tight clusters and drown the
signal. Phase 3 measures this with a noise-only control; until then, treat clustering
quality as unproven.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field

import numpy as np

from .models import DataPoint, Hypothesis
from .scoring import Scorer

# |log lift| below this counts as "no hypothesis has any bearing on this data point"
DEFAULT_RESIDUAL_EPSILON = 0.15

ABDUCE_PROMPT = """The statements below were all observed in the same case. They are
currently unexplained: none of the explanations under consideration makes any of them
more or less expected.

{cluster_block}

Propose ONE single hypothesis that, if true, would make these statements expected —
that is, a hypothesis they would all be evidence FOR. Requirements:
- one specific, falsifiable proposition, not a summary of the statements
- it must account for the statements jointly, not just one of them
- state it as a claim about what is the case, in one sentence
- do not hedge, do not list alternatives

Return only JSON: {"hypothesis": "...", "why": "one short sentence"}"""


@dataclass
class ResidualCluster:
    dp_ids: list[str]
    coherence: float                  # mean pairwise cosine similarity within cluster
    abduced_hypothesis: str = ""
    abduction_why: str = ""
    explains_fraction: float = 0.0    # of cluster members the abduced H actually lifts
    admitted: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.dp_ids)


def residual_scores(hyps: list[Hypothesis], dps: list[DataPoint],
                    cells: dict) -> dict:
    """dp_id -> max_H |log lift(D,H)|. Near zero means no hypothesis bears on it.
    Pure arithmetic over the cached grid; no LLM calls."""
    out = {}
    for dp in dps:
        vals = [abs(cells[(h.hyp_id, dp.dp_id)].log_lift)
                for h in hyps if (h.hyp_id, dp.dp_id) in cells]
        out[dp.dp_id] = max(vals) if vals else 0.0
    return out


def find_residuals(hyps: list[Hypothesis], dps: list[DataPoint], cells: dict,
                   epsilon: float = DEFAULT_RESIDUAL_EPSILON) -> list[DataPoint]:
    scores = residual_scores(hyps, dps, cells)
    return [dp for dp in dps if scores.get(dp.dp_id, 0.0) <= epsilon]


def cluster_residuals(residuals: list[DataPoint], backend,
                      min_cluster_size: int = 3,
                      small_corpus_threshold: int = 20,
                      distance_threshold: float = 0.6) -> list[ResidualCluster]:
    """Cluster residuals by MUTUAL coherence (never by relevance to hypotheses).

    Two algorithms, chosen by size — this is not a preference but a hard constraint:
    HDBSCAN returns ZERO clusters below roughly 20 points even when a perfectly
    separated cluster is present (density estimation has too little to work with).
    Measured: at n=10 with a tight planted cluster, HDBSCAN finds nothing while
    agglomerative clustering recovers it cleanly.

      n < small_corpus_threshold : agglomerative, cosine distance threshold
      n >= small_corpus_threshold: HDBSCAN (no preset cluster count; treats most
                                   points as noise — the right prior when hunting a
                                   small planted signal in a large corpus)

    NOTE: the embedding backend matters enormously here. `hashing` is lexical and will
    not produce meaningful semantic coherence; real use needs sentence_transformers.
    """
    if len(residuals) < min_cluster_size:
        return []
    emb = backend.encode([dp.text for dp in residuals])

    if len(residuals) < small_corpus_threshold:
        from sklearn.cluster import AgglomerativeClustering
        labels = AgglomerativeClustering(
            n_clusters=None, distance_threshold=distance_threshold,
            metric="cosine", linkage="average").fit_predict(emb)
    else:
        from sklearn.cluster import HDBSCAN
        labels = HDBSCAN(min_cluster_size=min_cluster_size,
                         metric="euclidean").fit_predict(emb)

    clusters = []
    for lab in sorted(set(labels)):
        if lab == -1:                       # HDBSCAN noise label
            continue
        idx = [i for i, l in enumerate(labels) if l == lab]
        if len(idx) < min_cluster_size:
            continue
        sub = emb[idx]
        # mean pairwise cosine similarity (vectors are L2-normalised by the backends)
        sims = sub @ sub.T
        n = len(idx)
        coherence = float((sims.sum() - n) / max(n * (n - 1), 1))
        clusters.append(ResidualCluster(
            dp_ids=[residuals[i].dp_id for i in idx],
            coherence=coherence))
    clusters.sort(key=lambda c: -c.coherence)
    return clusters


def abduce_hypothesis(cluster: ResidualCluster, dp_by_id: dict, llm) -> ResidualCluster:
    """One LLM call: propose the hypothesis that would explain this cluster."""
    block = "\n".join(f"  - {dp_by_id[i].text}" for i in cluster.dp_ids
                      if i in dp_by_id)
    r = llm.json(ABDUCE_PROMPT.replace("{cluster_block}", block), max_tokens=1000)
    cluster.abduced_hypothesis = str(r.get("hypothesis", "")).strip()
    cluster.abduction_why = str(r.get("why", "")).strip()
    return cluster


def verify_abduction(cluster: ResidualCluster, dp_by_id: dict, scorer: Scorer,
                     min_fraction: float = 0.5,
                     support_threshold: float = 1.5) -> ResidualCluster:
    """Score the abduced hypothesis back against its own cluster.

    An abduced hypothesis is admitted only if it actually LIFTS a sufficient fraction
    of the cluster — this is the guard against the model inventing a plausible-sounding
    claim that doesn't explain the evidence. Phase-2 evaluation reports the
    false-hypothesis rate on COMPLETE hypothesis sets; a detector that constantly
    proposes hypotheses will be switched off by users.
    """
    if not cluster.abduced_hypothesis:
        return cluster
    h_new = Hypothesis.make(cluster.abduced_hypothesis)
    lifted = 0
    for dp_id in cluster.dp_ids:
        dp = dp_by_id.get(dp_id)
        if dp is None:
            continue
        if scorer.cell(h_new, dp).lift >= support_threshold:
            lifted += 1
    cluster.explains_fraction = lifted / max(len(cluster.dp_ids), 1)
    cluster.admitted = cluster.explains_fraction >= min_fraction
    if not cluster.admitted:
        cluster.notes.append(
            f"rejected: explains only {cluster.explains_fraction:.0%} of cluster")
    return cluster


def detect_missing_hypotheses(hyps: list[Hypothesis], dps: list[DataPoint], cells: dict,
                              scorer: Scorer, backend,
                              epsilon: float = DEFAULT_RESIDUAL_EPSILON,
                              min_cluster_size: int = 3,
                              distance_threshold: float = 0.6,
                              verify: bool = True,
                              progress=None) -> list[ResidualCluster]:
    """Full capability (b): residuals -> coherence clusters -> abduced hypotheses."""
    residuals = find_residuals(hyps, dps, cells, epsilon)
    if progress:
        progress(f"{len(residuals)} residual data points (of {len(dps)})")
    clusters = cluster_residuals(residuals, backend, min_cluster_size,
                                 distance_threshold=distance_threshold)
    if progress:
        progress(f"{len(clusters)} coherent residual cluster(s)")
    dp_by_id = {dp.dp_id: dp for dp in dps}
    out = []
    for c in clusters:
        # One failing abduction must not abort the whole request. Record the error on
        # the cluster and continue; a 500 tells the caller nothing.
        try:
            c = abduce_hypothesis(c, dp_by_id, scorer.llm)
            if verify:
                c = verify_abduction(c, dp_by_id, scorer)
        except Exception as e:
            c.notes.append(f"abduction failed: {type(e).__name__}: {e}")
            c.admitted = False
        if progress:
            progress(f"cluster({c.size}) coh={c.coherence:.2f} "
                     f"admitted={c.admitted} -> {c.abduced_hypothesis[:60]}")
        out.append(c)
    return out


# --------------------------------------------------------------- retrieval baseline
def retrieval_baseline(hyps: list[Hypothesis], dps: list[DataPoint], backend,
                       top_k: int | None = None) -> list[str]:
    """BASELINE for evaluation: rank the corpus by similarity to the EXISTING
    hypotheses and return the top-k data point ids.

    The structural argument is that retrieval cannot surface evidence for a hypothesis
    nobody entered. Phase-2 evaluation checks whether the planted missing-hypothesis
    cluster appears here; by construction it should not.

    top_k defaults to a THIRD of the corpus (min 3). A fixed top_k larger than the
    corpus would return everything and make the baseline trivially perfect — which
    would understate the capability rather than test it.
    """
    if not hyps or not dps:
        return []
    if top_k is None:
        top_k = max(3, len(dps) // 3)
    top_k = min(top_k, len(dps))
    # Encode hypotheses and data points in ONE call: fit-per-batch backends (tfidf)
    # build a different vocabulary per call, so separate encodes are not comparable.
    texts = [h.text for h in hyps] + [dp.text for dp in dps]
    emb = backend.encode(texts)
    h_emb = emb[:len(hyps)]
    d_emb = emb[len(hyps):]
    sims = d_emb @ h_emb.T                       # (n_dps, n_hyps)
    best = sims.max(axis=1)
    order = np.argsort(-best)[:top_k]
    return [dps[i].dp_id for i in order]
