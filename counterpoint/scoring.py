"""The scorer — the core primitive everything else reads from.

  lift(H, D) = P(D | H) / P(D)     >1 supports · <1 disfavours · ~1 inert

Design points, all settled empirically in the Tier-1 oracle experiments:
  * elicit the RATIO directly via an ordinal change label -> multiplier
    (never absolute-prob / prior, which inverts lift at extreme priors);
  * elicit relatedness in a SEPARATE call (mixing the two questions collapsed
    72% of cells to "unchanged" and contaminated the logical axis);
  * present each pair NEUTRALLY (no privileged "incident"), to avoid the
    explaining-away collider;
  * cache every (H, D, model) cell and every (D, model) prior — re-scoring free.
"""
from __future__ import annotations
import math
from .models import DataPoint, Hypothesis, LiftCell
from .config import Settings, LIFT_SCALE, GUARANTEED_PCH, LIFT_CAP, verdict_for
from .llm import LLM
from .store import Store

PRIOR_PROMPT = """Consider a randomly chosen case of this kind, knowing nothing specific
about any particular incident.

Statement: {stmt}

Give P(statement is true) as a base rate — how often a statement like this holds for
such a case in general. Do not reason about any specific investigation.

Return only JSON: {"prior": 0.0}  (strictly between 0 and 1)"""

REL_PROMPT = """How topically related are these two statements? Subject matter ONLY —
say nothing about whether one supports, implies, or contradicts the other.

A: {h}
B: {d}

Return only JSON: {"relatedness": 0.0}  (0 = unrelated, 1 = same subject)"""

LIFT_PROMPT = """Compare two scenarios.
  Scenario 1: you know only that this is a case of the general kind described.
  Scenario 2: you additionally learn that this is TRUE -- {h}

Now consider this statement: {d}

How does moving from Scenario 1 to Scenario 2 change the probability of that statement?
  "ruled_out"  Scenario 2 makes it essentially impossible
  "much_less"  about 5x less likely
  "less"       about 2x less likely
  "unchanged"  NO CHANGE AT ALL — learning the Scenario-2 fact tells you nothing about
               this statement. Choose this whenever there is no logical bearing, EVEN IF
               the two share subject matter.
  "more"       about 2x more likely
  "much_more"  about 5x more likely
  "guaranteed" Scenario 2 makes it essentially certain

Judge ONLY the change. Do not judge whether either statement is actually true.

Return only JSON: {"change": "...", "why": "one short sentence"}"""


def _lift_from_change(change: str, prior: float) -> float:
    mult = LIFT_SCALE.get(change, 1.0)
    if mult is None:                      # "guaranteed"
        p_cond = GUARANTEED_PCH
        lift = p_cond / max(prior, 1e-3)
    else:
        lift = mult
    return float(min(max(lift, 1e-3), LIFT_CAP))


def _safe_float(value, default: float) -> float:
    """Coerce a model-supplied value to float, falling back on anything odd
    (strings like 'high', None, NaN, lists). LLMs occasionally ignore the schema;
    the scorer must degrade gracefully rather than crash a whole grid."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if f != f or f in (float("inf"), float("-inf")):   # NaN / inf
        return default
    return f


class Scorer:
    def __init__(self, llm: LLM, store: Store, settings: Settings):
        self.llm, self.store, self.s = llm, store, settings
        self.model = getattr(llm, "model", "unknown")

    def prior(self, dp: DataPoint) -> float:
        cached = self.store.get_prior(dp.dp_id, self.model)
        if cached is not None:
            return cached
        r = self.llm.json(PRIOR_PROMPT.replace("{stmt}", dp.text), max_tokens=120)
        p = float(min(max(_safe_float(r.get("prior"), 0.5), 1e-3), 0.99))
        self.store.put_prior(dp.dp_id, self.model, p)
        return p

    def cell(self, h: Hypothesis, dp: DataPoint) -> LiftCell:
        cached = self.store.get_cell(h.hyp_id, dp.dp_id, self.model)
        if cached is not None:
            return cached
        prior = self.prior(dp)
        rel = min(max(_safe_float(self.llm.json(
            REL_PROMPT.replace("{h}", h.text).replace("{d}", dp.text), max_tokens=60
        ).get("relatedness"), 0.0), 0.0), 1.0)
        lr = self.llm.json(
            LIFT_PROMPT.replace("{h}", h.text).replace("{d}", dp.text), max_tokens=200)
        change = str(lr.get("change", "unchanged")).strip().lower()
        if change not in LIFT_SCALE:
            change = "unchanged"
        lift = _lift_from_change(change, prior)
        log_lift = math.log(lift)
        cell = LiftCell(hyp_id=h.hyp_id, dp_id=dp.dp_id, change=change, lift=lift,
                        log_lift=log_lift, relatedness=rel, verdict=verdict_for(log_lift),
                        rationale=str(lr.get("why", "")), model=self.model)
        self.store.put_cell(cell)
        return cell

    def grid(self, hyps: list[Hypothesis], dps: list[DataPoint],
             progress=None) -> dict:
        cells = {}
        total = len(hyps) * len(dps)
        i = 0
        for h in hyps:
            for dp in dps:
                cells[(h.hyp_id, dp.dp_id)] = self.cell(h, dp)
                i += 1
                if progress:
                    progress(i, total)
        return cells
