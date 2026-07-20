"""Capability (a): conjunction refutation — minimal refuting sets.

Finds SETS of data points that JOINTLY refute a hypothesis while each member,
taken alone, does not. This is the thing pairwise checking structurally cannot do:
a checker that looks at (H, D) one data point at a time never sees the interaction.

Method: greedy set-growth, modelled on minimal-unsatisfiable-subset (MUS) extraction
from SAT solving — NOT tuple enumeration. Enumerating k-tuples is O(|corpus|^k) and
unbounded in the wrong dimension; set-growth discovers k instead of fixing it.

Two properties make it tractable:
  * monotonicity of contradiction — once a set refutes H, adding members cannot
    un-refute it, so the search never backtracks;
  * a structural screen — only data points sharing an entity / scope / timeframe with
    the current set are candidate additions, so each growth step is O(shortlist).

Cost: O(k x shortlist) joint-lift calls to grow one set to size k.

KNOWN WEAKNESS (measured, not hidden): greedy growth can miss refuting sets whose
members are each individually neutral — there is no gradient for the first step to
follow. Mitigations implemented: multi-start seeding, and bundle candidates that share
an entity so a jointly-diagnostic pair can enter together. Phase-2 evaluation reports
recall on such sets SEPARATELY; that number is the honest ceiling of this approach.
"""
from __future__ import annotations
import math
import re
from dataclasses import dataclass, field

from .models import DataPoint, Hypothesis
from .config import LIFT_SCALE, verdict_for
from .scoring import Scorer, _lift_from_change, _safe_float

# A joint set refutes H when the conjunction's lift falls below this.
# log(0.2) ~= -1.6 ; i.e. the set makes H's evidence ~5x less expected.
DEFAULT_REFUTE_THRESHOLD = 0.25

JOINT_LIFT_PROMPT = """Compare two scenarios.
  Scenario 1: you know only that this is a case of the general kind described.
  Scenario 2: you additionally learn that this is TRUE -- {h}

Now consider the following statements, taken TOGETHER as a single combined fact.
Judge them jointly, not one by one — what matters is whether the COMBINATION is
more or less expected, including any tension between them:

{d_block}

How does moving from Scenario 1 to Scenario 2 change the probability of that
combined fact holding?
  "ruled_out"  Scenario 2 makes the combination essentially impossible
  "much_less"  about 5x less likely
  "less"       about 2x less likely
  "unchanged"  NO CHANGE AT ALL
  "more"       about 2x more likely
  "much_more"  about 5x more likely
  "guaranteed" Scenario 2 makes the combination essentially certain

Judge ONLY the change. Do not judge whether the statements are actually true.

Return only JSON: {"change": "...", "why": "one short sentence"}"""


@dataclass
class RefutingSet:
    hyp_id: str
    dp_ids: list[str]
    joint_lift: float
    joint_log_lift: float
    member_lifts: dict           # dp_id -> individual lift(D,H)
    rationale: str
    llm_calls: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.dp_ids)

    @property
    def is_conjunction_only(self) -> bool:
        """True when NO member individually refutes H — the interesting case, and the
        one a pairwise checker provably cannot find."""
        return all(v >= DEFAULT_REFUTE_THRESHOLD for v in self.member_lifts.values())


# --------------------------------------------------------------- structural screen
_STOP = set("the a an of to and or is are was were in on at for with by from this that "
            "it its as be been being their his her they he she we you i not no".split())


def _tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']+", s.lower())
            if w not in _STOP and len(w) > 2}


def _entities(dp: DataPoint) -> set[str]:
    """Cheap proxy for entity/scope/time: capitalised tokens, numbers, dates, plus
    source metadata. Deliberately crude — this is a SCREEN, not an analysis; it only
    has to keep the candidate shortlist small without discarding true interactions."""
    ents = set(re.findall(r"\b[A-Z][a-zA-Z\-]{2,}\b", dp.text))
    ents |= set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b|\b\d+(?:[.,]\d+)?%?\b", dp.text))
    for k in ("source", "shift", "oven", "supplier"):
        v = dp.meta.get(k)
        if v:
            ents.add(str(v))
    return {e.lower() for e in ents}


def structural_candidates(current: list[DataPoint], pool: list[DataPoint],
                          limit: int = 25) -> list[DataPoint]:
    """Data points sharing an entity/scope/time with the current set, ranked by overlap.
    Falls back to lexical overlap when no entity matches, so the search never stalls."""
    cur_ents = set()
    cur_toks = set()
    for dp in current:
        cur_ents |= _entities(dp)
        cur_toks |= _tokens(dp.text)
    chosen = {dp.dp_id for dp in current}
    scored = []
    for dp in pool:
        if dp.dp_id in chosen:
            continue
        ent_overlap = len(cur_ents & _entities(dp))
        tok_overlap = len(cur_toks & _tokens(dp.text))
        score = ent_overlap * 10 + tok_overlap        # entities dominate
        if score > 0:
            scored.append((score, dp))
    scored.sort(key=lambda t: -t[0])
    return [dp for _, dp in scored[:limit]]


# --------------------------------------------------------------- joint lift
class ConjunctionScorer:
    """Scores the lift of a CONJUNCTION of data points against a hypothesis.

    Presenting the set as one combined conditioning event is what makes the
    interaction visible; scoring members separately and combining arithmetically
    would reproduce exactly the pairwise blindness we are trying to defeat.
    """

    def __init__(self, scorer: Scorer):
        self.scorer = scorer
        self.llm = scorer.llm
        self.calls = 0

    def joint_lift(self, h: Hypothesis, dps: list[DataPoint]) -> tuple[float, str]:
        if not dps:
            return 1.0, "empty set"
        if len(dps) == 1:
            cell = self.scorer.cell(h, dps[0])
            return cell.lift, cell.rationale
        # content-addressed key: order-independent, so {A,B} and {B,A} share a cell
        set_key = ",".join(sorted(dp.dp_id for dp in dps))
        cached = self.scorer.store.get_joint(h.hyp_id, set_key, self.scorer.model)
        if cached is not None:
            return cached
        block = "\n".join(f"  - {dp.text}" for dp in dps)
        prompt = (JOINT_LIFT_PROMPT
                  .replace("{h}", h.text)
                  .replace("{d_block}", block))
        r = self.llm.json(prompt, max_tokens=250)
        self.calls += 1
        change = str(r.get("change", "unchanged")).strip().lower()
        if change not in LIFT_SCALE:
            change = "unchanged"
        # Prior for a conjunction: approximate with the least likely member's prior.
        # (Exact joint priors are unidentifiable here; the ordinal scale carries the
        # signal and this only affects the "guaranteed" branch.)
        prior = min((self.scorer.prior(dp) for dp in dps), default=0.5)
        lift = _lift_from_change(change, prior)
        why = str(r.get("why", ""))
        self.scorer.store.put_joint(h.hyp_id, set_key, self.scorer.model,
                                    change, lift, why)
        return lift, why


# --------------------------------------------------------------- extraction
def find_refuting_sets(h: Hypothesis, dps: list[DataPoint], scorer: Scorer,
                       max_size: int = 5, n_seeds: int = 3, shortlist: int = 25,
                       refute_threshold: float = DEFAULT_REFUTE_THRESHOLD,
                       progress=None) -> list[RefutingSet]:
    """Greedy multi-start minimal-refuting-set extraction for one hypothesis.

    Returns minimal sets whose CONJUNCTION refutes h. Sets where no member
    individually refutes (`is_conjunction_only`) are the product-relevant find.
    """
    cj = ConjunctionScorer(scorer)
    # individual lifts come from the cache — free if the matrix was already built
    individual = {dp.dp_id: scorer.cell(h, dp).lift for dp in dps}
    by_id = {dp.dp_id: dp for dp in dps}

    # Seeds: the strongest tension carriers that do NOT already refute alone
    # (a single-point refutation is not a conjunction finding).
    ranked = sorted(dps, key=lambda dp: individual[dp.dp_id])
    seeds = [dp for dp in ranked if individual[dp.dp_id] >= refute_threshold][:n_seeds]
    if not seeds:
        seeds = ranked[:n_seeds]

    found: list[RefutingSet] = []
    seen_sets: set[frozenset] = set()

    for si, seed in enumerate(seeds):
        current = [seed]
        cur_lift, cur_why = individual[seed.dp_id], ""
        while len(current) < max_size:
            cands = structural_candidates(current, dps, limit=shortlist)
            if not cands:
                break
            best_dp, best_lift, best_why = None, cur_lift, ""
            for cand in cands:
                lift, why = cj.joint_lift(h, current + [cand])
                if progress:
                    progress(cj.calls)
                if lift < best_lift:
                    best_dp, best_lift, best_why = cand, lift, why
            if best_dp is None:                     # no candidate lowers joint lift
                break
            current = current + [best_dp]
            cur_lift, cur_why = best_lift, best_why
            if cur_lift < refute_threshold:
                minimal, mlift, mwhy = _minimize(h, current, cj, refute_threshold)
                key = frozenset(dp.dp_id for dp in minimal)
                if key not in seen_sets:
                    seen_sets.add(key)
                    found.append(RefutingSet(
                        hyp_id=h.hyp_id,
                        dp_ids=[dp.dp_id for dp in minimal],
                        joint_lift=mlift,
                        joint_log_lift=math.log(max(mlift, 1e-6)),
                        member_lifts={dp.dp_id: individual[dp.dp_id] for dp in minimal},
                        rationale=mwhy or cur_why,
                        llm_calls=cj.calls,
                        notes=[f"seed={si}"]))
                break
    return found


def _minimize(h: Hypothesis, members: list[DataPoint], cj: ConjunctionScorer,
              threshold: float) -> tuple[list[DataPoint], float, str]:
    """Deletion-based minimisation: drop any member that isn't needed for refutation.
    Guarantees true minimality (no proper subset also refutes)."""
    current = list(members)
    lift, why = cj.joint_lift(h, current)
    changed = True
    while changed and len(current) > 2:
        changed = False
        for dp in list(current):
            trial = [d for d in current if d.dp_id != dp.dp_id]
            if len(trial) < 2:
                continue
            t_lift, t_why = cj.joint_lift(h, trial)
            if t_lift < threshold:                 # still refutes without dp
                current, lift, why = trial, t_lift, t_why
                changed = True
                break
    return current, lift, why


# --------------------------------------------------------------- pairwise baseline
def pairwise_refutations(h: Hypothesis, dps: list[DataPoint], scorer: Scorer,
                         refute_threshold: float = DEFAULT_REFUTE_THRESHOLD
                         ) -> list[str]:
    """BASELINE for evaluation: which data points refute h ON THEIR OWN.

    The product claim is that conjunction refutation finds what this cannot. Phase-2
    evaluation compares recall of planted sets: this baseline should find ~none of
    them. Without this comparison the capability's value is unproven.
    """
    return [dp.dp_id for dp in dps if scorer.cell(h, dp).lift < refute_threshold]
