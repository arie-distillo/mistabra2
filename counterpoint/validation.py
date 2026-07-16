"""Assemble the validation matrix from cached lift cells. Pure arithmetic — no
LLM calls. This is the read-model behind the matrix UI and the hypothesis-detail
view.

Two derived quantities:
  * hypothesis SCORE = sum of credibility-weighted log-lift over all data points.
    Support pushes up, contradiction pushes down, inert data points contribute ~0.
  * data-point DIAGNOSTICITY = variance of log-lift across hypotheses. High = the
    data point discriminates between hypotheses; ~0 = it fits them all equally and
    is analytically worthless (this reproduces Heuer's "strike non-diagnostic rows"
    as a computation). It is also the seed of residual detection in later steps.

NOTE ON SCOPE: this deliberately does NOT implement ACH elimination / worst-cell
verdict logic. The product is a graded validation engine, not an ACH tool. The
worst/best cells are surfaced as evidence, not used to eliminate hypotheses.
"""
from __future__ import annotations
import numpy as np
from .models import Hypothesis, DataPoint, LiftCell, HypothesisResult, Matrix
from .config import verdict_for


def _credibility(dp: DataPoint) -> float:
    c = dp.meta.get("credibility")
    try:
        return float(c) if c is not None else 1.0
    except (TypeError, ValueError):
        return 1.0


def hypothesis_result(h: Hypothesis, dps: list[DataPoint],
                      cells: dict) -> HypothesisResult:
    my = [cells[(h.hyp_id, dp.dp_id)] for dp in dps if (h.hyp_id, dp.dp_id) in cells]
    cred = {dp.dp_id: _credibility(dp) for dp in dps}
    score = sum(c.log_lift * cred.get(c.dp_id, 1.0) for c in my)
    worst = min(my, key=lambda c: c.log_lift) if my else None
    best = max(my, key=lambda c: c.log_lift) if my else None
    n_sup = sum(c.verdict in ("supports", "strongly_supports") for c in my)
    n_con = sum(c.verdict in ("contradicts", "strongly_contradicts") for c in my)
    n_neu = len(my) - n_sup - n_con
    return HypothesisResult(
        hypothesis=h, cells=my, score=score, worst_cell=worst, best_cell=best,
        verdict=verdict_for(score / max(len(my), 1)),   # mean-log-lift band
        n_supporting=n_sup, n_contradicting=n_con, n_neutral=n_neu)


def diagnosticity(hyps: list[Hypothesis], dps: list[DataPoint],
                  cells: dict) -> dict:
    out = {}
    for dp in dps:
        lls = [cells[(h.hyp_id, dp.dp_id)].log_lift
               for h in hyps if (h.hyp_id, dp.dp_id) in cells]
        out[dp.dp_id] = float(np.var(lls)) if lls else 0.0
    return out


def build_matrix(hyps: list[Hypothesis], dps: list[DataPoint], cells: dict) -> Matrix:
    results = [hypothesis_result(h, dps, cells) for h in hyps]
    results.sort(key=lambda r: -r.score)
    return Matrix(hypotheses=hyps, data_points=dps, cells=cells, results=results,
                  diagnosticity=diagnosticity(hyps, dps, cells))
