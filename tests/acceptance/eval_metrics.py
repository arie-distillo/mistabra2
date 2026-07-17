"""Correctness evaluation over HTTP — the `--eval` layer of the acceptance runner.

Unlike the machinery journeys (which assert the system WORKS), this measures whether
the system's answers are CORRECT, by comparing the scored matrix against each
scenario's ground_truth. Everything here is computed from API responses; the only
local input is the ground_truth block from the bundled YAML (the server never sees it).

Output is GRADED MEASUREMENTS, not pass/fail. Per the agreed design, these metrics are
reported but never change the run's exit code — a metric riding on one model's
calibration must not gate a deploy.

Three directional metrics per scenario:
  M1 surviving-not-last : the hypothesis ground truth marks `is_surviving_hypothesis`
                          is not the lowest-scored.
  M2 eliminated-low     : hypotheses ground truth marks `eliminated: true` score below
                          the scenario mean. (fraction correct)
  M3 noise-below-real   : injected noise data points have lower mean diagnosticity than
                          real evidence.
"""
from __future__ import annotations
import statistics
from dataclasses import dataclass, field

# ground_truth is read locally from the same bundled scenarios the server loads
from counterpoint.fixtures import all_scenarios


@dataclass
class ScenarioMetrics:
    index: int
    name: str
    m1: float | None
    m2: float | None
    m3: float | None
    top_hypothesis: str
    notes: list[str] = field(default_factory=list)


def _ground_truth(index: int) -> list[dict]:
    """The ordered ground-truth claim records for a scenario (H1..Hn order)."""
    return list(all_scenarios()[index].ground_truth.get("claims", []))


def compute_metrics(index: int, matrix_json: dict) -> ScenarioMetrics:
    """Compute M1/M2/M3 for one scenario from its /api/matrix response.

    matrix_json is the parsed body of POST /api/matrix:
      { "hypotheses": [{hyp_id, text, score, ...}, ...],   # ranked desc by score
        "data_points": [{dp_id, text, diagnosticity, noise, ...}, ...], ... }
    """
    hyps = matrix_json.get("hypotheses", [])
    dps = matrix_json.get("data_points", [])
    gt = _ground_truth(index)

    # Map ground-truth records to the server's hypotheses BY TEXT. The YAML claims are
    # in H1..Hn order; the API returns hypotheses ranked by score with their text, so we
    # align on text rather than on server-assigned ids.
    gt_by_text = {}
    scenario = all_scenarios()[index]
    for i, claim in enumerate(scenario.hypotheses):
        if i < len(gt):
            gt_by_text[claim["text"].strip()] = gt[i]

    scores = {h["text"].strip(): h["score"] for h in hyps}
    ranked_texts = [h["text"].strip() for h in hyps]           # already score-desc
    mean_score = statistics.mean(scores.values()) if scores else 0.0
    notes: list[str] = []

    # M1: surviving hypothesis not ranked last
    m1 = None
    surviving = [t for t, g in gt_by_text.items() if g.get("is_surviving_hypothesis")]
    if surviving and ranked_texts:
        last = ranked_texts[-1]
        m1 = 1.0 if all(t != last for t in surviving) else 0.0
        if m1 == 0.0:
            notes.append("surviving hypothesis ranked last")

    # M2: eliminated hypotheses below the mean score
    m2 = None
    elim = [t for t, g in gt_by_text.items() if g.get("eliminated")]
    if elim:
        below = sum(scores.get(t, 0.0) < mean_score for t in elim)
        m2 = below / len(elim)

    # M3: injected noise less diagnostic than real evidence
    noise_d = [dp["diagnosticity"] for dp in dps if dp.get("noise") == "irrelevant"]
    real_d = [dp["diagnosticity"] for dp in dps if dp.get("noise") != "irrelevant"]
    m3 = None
    if noise_d and real_d:
        m3 = 1.0 if statistics.mean(noise_d) <= statistics.mean(real_d) else 0.0
        if m3 == 0.0:
            notes.append("noise more diagnostic than real evidence")

    top = ranked_texts[0] if ranked_texts else ""
    return ScenarioMetrics(index=index, name=scenario.name, m1=m1, m2=m2, m3=m3,
                           top_hypothesis=top, notes=notes)


def _fmt(x):
    return "  —  " if x is None else f"{x:5.2f}"


def print_metrics_report(results: list[ScenarioMetrics], model_label: str, log):
    """Print the graded-measurement table. Uses the runner's logger so it also lands
    in --log-file. This is a REPORT, not a verdict — it never sets the exit code."""
    log.always("")
    log.always(f"{'model':28s}{'M1':>6s}{'M2':>6s}{'M3':>6s}  scenario")
    log.always("-" * 86)
    agg = {"M1": [], "M2": [], "M3": []}
    for r in results:
        log.always(f"{model_label[:27]:28s}{_fmt(r.m1)}{_fmt(r.m2)}{_fmt(r.m3)}  {r.name[:38]}")
        if r.notes:
            log.always(f"{'':28s}      ! {'; '.join(r.notes)}")
        for k, v in (("M1", r.m1), ("M2", r.m2), ("M3", r.m3)):
            if v is not None:
                agg[k].append(v)
    log.always("-" * 86)
    means = {k: (statistics.mean(v) if v else None) for k, v in agg.items()}
    log.always(f"{'AGGREGATE':28s}{_fmt(means['M1'])}{_fmt(means['M2'])}{_fmt(means['M3'])}")
    log.always("")
    log.always("M1 surviving-not-last · M2 eliminated-below-mean · M3 noise-below-real")
    log.always("Graded measurements, not pass/fail — tracked over time and across models.")
    log.always("These do NOT affect the run's exit code; the pass/fail verdict is the")
    log.always("machinery journeys above.")
    return means
