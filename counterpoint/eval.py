"""Step 5 — evaluation harness (quality, not pass/fail).

This is the productised Tier-1 experiment. It runs the REAL scorer over the
bundled scenarios and measures DIRECTIONAL quality against each scenario's
`ground_truth` block (which the scorer itself never sees). Quality is graded, so
these are METRICS WITH TOLERANCES reported over time — not hard assertions that
gate a build. Conflating the two is what makes an LLM test suite flaky.

Three directional metrics per scenario:
  M1 surviving-not-last : the hypothesis ground truth marks `is_surviving_hypothesis`
                          is not the lowest-scored. (recall-style, per scenario)
  M2 eliminated-low     : hypotheses ground truth marks `eliminated: true` score
                          below the scenario mean. (fraction correct)
  M3 noise-below-real   : injected noise data points have lower mean diagnosticity
                          than real evidence. (boolean per scenario)

Run as a script (recommended — prints a report):
    python -m counterpoint.eval
    python -m counterpoint.eval --models anthropic/claude-sonnet-4.5,openai/gpt-5
    python -m counterpoint.eval --scenario 0 --noise 5

Run as a gated pytest (asserts only that metrics clear conservative floors):
    python -m pytest -m eval --run-live
"""
from __future__ import annotations
import argparse
import statistics
from dataclasses import dataclass, field

from .config import Settings
from .store import Store
from .pipeline import Pipeline
from .llm import MockLLM
from .fixtures import all_scenarios, inject_noise, load_into_pipeline


@dataclass
class ScenarioEval:
    name: str
    model: str
    m1_surviving_not_last: float | None      # 1.0 / 0.0 / None if gt absent
    m2_eliminated_low_frac: float | None
    m3_noise_below_real: float | None
    top_hypothesis: str
    notes: list[str] = field(default_factory=list)


def _pipeline_for(model: str | None) -> Pipeline:
    s = Settings(cache_db_path=":memory:", embedding_backend="hashing")
    if model:
        s.model = model
    llm = None                     # let Pipeline choose: real if key, else mock
    if not s.has_api_key:
        llm = MockLLM()
    return Pipeline(s, llm=llm, store=Store(":memory:"))


def evaluate_scenario(index: int, model: str | None = None,
                      noise: int = 4) -> ScenarioEval:
    scenario = all_scenarios()[index]
    gt = {c["id"]: c for c in scenario.ground_truth.get("claims", [])}
    # align ground-truth ids to hypotheses by ORDER (yaml claims are in H1..Hn order)
    scenario_noisy = inject_noise(scenario, n=noise, duplicate=0, seed=index)

    pipe = _pipeline_for(model)
    hyps = load_into_pipeline(pipe, scenario_noisy)
    m = pipe.matrix()

    # map each hypothesis (by order of addition) to its ground-truth record
    gt_list = list(scenario.ground_truth.get("claims", []))
    order_to_gt = {h.hyp_id: gt_list[i] for i, h in enumerate(hyps) if i < len(gt_list)}

    ranked = m.results                             # already sorted desc by score
    ranked_ids = [r.hypothesis.hyp_id for r in ranked]
    scores = {r.hypothesis.hyp_id: r.score for r in ranked}
    mean_score = statistics.mean(scores.values()) if scores else 0.0

    notes = []

    # M1: surviving hypothesis not last
    m1 = None
    surviving = [hid for hid, g in order_to_gt.items()
                 if g.get("is_surviving_hypothesis")]
    if surviving:
        last = ranked_ids[-1]
        m1 = 1.0 if all(s != last for s in surviving) else 0.0
        if m1 == 0.0:
            notes.append("surviving hypothesis ranked last")

    # M2: eliminated hypotheses below mean
    m2 = None
    elim = [hid for hid, g in order_to_gt.items() if g.get("eliminated")]
    if elim:
        below = sum(scores.get(hid, 0.0) < mean_score for hid in elim)
        m2 = below / len(elim)

    # M3: noise diagnosticity below real
    dp_by = {dp.dp_id: dp for dp in m.data_points}
    noise_d, real_d = [], []
    for dp_id, d in m.diagnosticity.items():
        if dp_by[dp_id].meta.get("noise") == "irrelevant":
            noise_d.append(d)
        else:
            real_d.append(d)
    m3 = None
    if noise_d and real_d:
        m3 = 1.0 if statistics.mean(noise_d) <= statistics.mean(real_d) else 0.0
        if m3 == 0.0:
            notes.append("noise more diagnostic than real evidence")

    return ScenarioEval(
        name=scenario.name, model=pipe.scorer.model,
        m1_surviving_not_last=m1, m2_eliminated_low_frac=m2,
        m3_noise_below_real=m3,
        top_hypothesis=ranked[0].hypothesis.text if ranked else "",
        notes=notes)


def run_eval(models: list[str] | None, scenario: int | None, noise: int) -> list[ScenarioEval]:
    models = models or [None]
    idxs = [scenario] if scenario is not None else range(len(all_scenarios()))
    out = []
    for mdl in models:
        for i in idxs:
            out.append(evaluate_scenario(i, mdl, noise))
    return out


def _fmt(x):
    return "  —  " if x is None else f"{x:5.2f}"


def print_report(results: list[ScenarioEval]):
    print(f"\n{'model':32s}{'M1':>6s}{'M2':>6s}{'M3':>6s}  scenario")
    print("-" * 90)
    agg = {"M1": [], "M2": [], "M3": []}
    for r in results:
        print(f"{r.model[:31]:32s}{_fmt(r.m1_surviving_not_last)}"
              f"{_fmt(r.m2_eliminated_low_frac)}{_fmt(r.m3_noise_below_real)}"
              f"  {r.name[:40]}")
        if r.notes:
            print(f"{'':32s}      ! {'; '.join(r.notes)}")
        for k, v in (("M1", r.m1_surviving_not_last), ("M2", r.m2_eliminated_low_frac),
                     ("M3", r.m3_noise_below_real)):
            if v is not None:
                agg[k].append(v)
    print("-" * 90)
    means = {k: (statistics.mean(v) if v else None) for k, v in agg.items()}
    print(f"{'AGGREGATE':32s}{_fmt(means['M1'])}{_fmt(means['M2'])}{_fmt(means['M3'])}")
    print("\nM1 surviving-not-last · M2 eliminated-below-mean · M3 noise-below-real")
    print("These are quality measurements, not pass/fail. Track them over time and")
    print("across models; a metric that flips between models is riding on one model's")
    print("calibration, not on the method.\n")
    return means


def main():
    ap = argparse.ArgumentParser(description="Counterpoint evaluation harness")
    ap.add_argument("--models", type=str, default="",
                    help="comma-separated OpenRouter model ids (default: env/mock)")
    ap.add_argument("--scenario", type=int, default=None, help="single scenario index")
    ap.add_argument("--noise", type=int, default=4)
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()] or None
    results = run_eval(models, args.scenario, args.noise)
    print_report(results)


if __name__ == "__main__":
    main()
