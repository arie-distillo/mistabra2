"""Unit tests: graceful degradation when the LLM returns malformed output.

The mock and the acceptance tests prove the plumbing when the model BEHAVES. This
file proves the scorer survives when the model MISBEHAVES — off-scale labels,
non-numeric priors, out-of-range values, empty responses — the one thing neither
the deterministic mock nor a real-LLM run reliably triggers, because you can't
schedule a model to emit garbage on cue.

Everything is offline: a fault-injecting fake LLM returns canned bad payloads and
we assert the scorer produces a valid cell instead of crashing a whole grid.
"""
import math
import pytest

from counterpoint.scoring import Scorer, _safe_float
from counterpoint.store import Store
from counterpoint.config import Settings, LIFT_CAP
from counterpoint.models import Hypothesis, DataPoint


# ---- the numeric-coercion helper the scorer relies on ----
@pytest.mark.parametrize("value,default,expected", [
    (0.5, 0.1, 0.5),
    ("0.7", 0.1, 0.7),
    ("high", 0.1, 0.1),          # non-numeric string -> default
    (None, 0.1, 0.1),
    ([0.5], 0.1, 0.1),           # wrong type -> default
    (float("nan"), 0.1, 0.1),
    (float("inf"), 0.1, 0.1),
])
def test_safe_float_coerces_or_defaults(value, default, expected):
    assert _safe_float(value, default) == expected


# ---- fault injection at the LLM boundary ----
class FaultyLLM:
    """Returns configured payloads regardless of the prompt, so we can drive the
    scorer with deliberately broken model output. `model` attr keys the cache."""
    model = "faulty/test"

    def __init__(self, prior=None, rel=None, lift=None):
        self._prior = prior if prior is not None else {"prior": 0.5}
        self._rel = rel if rel is not None else {"relatedness": 0.5}
        self._lift = lift if lift is not None else {"change": "unchanged"}

    def json(self, prompt, max_tokens=500):
        p = prompt.lower()
        if "base rate" in p:
            return self._prior
        if "topically related" in p:
            return self._rel
        return self._lift


def _score_one(llm):
    scorer = Scorer(llm, Store(":memory:"), Settings())
    h = Hypothesis.make("The flour is defective")
    dp = DataPoint.make("c", "d", "Only new-supplier batches were overbaked")
    return scorer.cell(h, dp)


def test_off_scale_change_label_falls_back_to_unchanged():
    cell = _score_one(FaultyLLM(lift={"change": "catastrophic", "why": "x"}))
    assert cell.change == "unchanged"
    assert cell.lift == 1.0


def test_missing_change_key_falls_back():
    cell = _score_one(FaultyLLM(lift={"why": "no change field at all"}))
    assert cell.change == "unchanged"


def test_non_numeric_prior_does_not_crash():
    cell = _score_one(FaultyLLM(prior={"prior": "very likely"}))
    assert math.isfinite(cell.log_lift)


def test_out_of_range_prior_is_clamped():
    # prior of 1.5 is illegal; must not blow up the guaranteed-branch division
    cell = _score_one(FaultyLLM(prior={"prior": 1.5}, lift={"change": "guaranteed"}))
    assert 1e-3 <= cell.lift <= LIFT_CAP


def test_non_numeric_relatedness_defaults_to_zero():
    cell = _score_one(FaultyLLM(rel={"relatedness": "kind of"}))
    assert cell.relatedness == 0.0


def test_relatedness_clamped_to_unit_interval():
    cell = _score_one(FaultyLLM(rel={"relatedness": 7.3}))
    assert 0.0 <= cell.relatedness <= 1.0


def test_empty_lift_response_is_survivable():
    cell = _score_one(FaultyLLM(lift={}))
    assert cell.change == "unchanged" and math.isfinite(cell.lift)
