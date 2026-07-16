"""Unit tests: the lift scale and verdict bands.

These cover the scoring ARITHMETIC in isolation — how an oracle's ordinal change
label becomes a lift multiplier, and how a log-lift becomes a verdict band. Pure
functions, no LLM, no I/O.

The most important test here is the regression guard that "much_less" never
inverts to a lift above 1 — the exact bug that made an early version silently
treat disfavouring evidence as support.
"""
from counterpoint.config import verdict_for, LIFT_CAP, LIFT_SCALE
from counterpoint.scoring import _lift_from_change


# ---- the lift scale: an ordinal label -> a multiplier ----
def test_much_less_is_below_one_regardless_of_prior():
    # regression: disfavouring evidence must never come out as support
    assert _lift_from_change("much_less", 0.05) < 1.0
    assert _lift_from_change("much_less", 0.90) < 1.0


def test_much_more_above_one_and_unchanged_is_one():
    assert _lift_from_change("much_more", 0.5) > 1.0
    assert _lift_from_change("unchanged", 0.3) == 1.0


def test_guaranteed_scales_inversely_with_prior():
    # a guaranteed RARE fact carries more lift than a guaranteed common one
    assert _lift_from_change("guaranteed", 0.05) > _lift_from_change("guaranteed", 0.5)


def test_lift_is_clamped_to_cap():
    assert _lift_from_change("guaranteed", 0.0001) <= LIFT_CAP


def test_every_label_stays_bounded():
    # exhaustively: no label at any prior escapes [1e-3, CAP]
    for label in LIFT_SCALE:
        for prior in (1e-4, 0.5, 0.999):
            v = _lift_from_change(label, prior)
            assert 1e-3 <= v <= LIFT_CAP


# ---- verdict bands: a log-lift -> a categorical verdict ----
def test_verdict_bands_cover_the_spectrum():
    assert verdict_for(-2.0) == "strongly_contradicts"
    assert verdict_for(-0.5) == "contradicts"
    assert verdict_for(0.0) == "neutral"
    assert verdict_for(0.5) == "supports"
    assert verdict_for(2.0) == "strongly_supports"
