"""Capability evaluation — runs authored test cases against the two capabilities.

This is the PROVING step (what the phase plan called Phase 2). Phase 1 built the
capabilities; this measures whether they work, against independently-authored cases
with planted ground truth.

Everything runs over HTTP, so it works against a local server or a deployment.

What it measures, per capability:

  (a) conjunction refutation
      set_recall      — was the planted refuting set found (as a set or a superset)?
      conjunction_only— did the found set consist of members that don't refute alone?
      false_positives — sets reported on NEGATIVE CONTROL cases (no refutation exists)
      pairwise_recall — BASELINE: what a pairwise-only checker finds. The capability's
                        value is the GAP between set_recall and pairwise_recall, not
                        the absolute number.

  (b) missing-hypothesis detection
      cluster_recall  — fraction of the planted cluster recovered
      cluster_purity  — fraction of the recovered cluster that is actually planted
      abduction       — the proposed hypothesis text (reported for judging, not scored
                        automatically: grading an abduction with the same model that
                        produced it is not evidence)
      false_hypothesis— hypotheses proposed on COMPLETE-set controls (hallucination)
      retrieval_recall— BASELINE: does similarity ranking against the GIVEN hypotheses
                        surface the planted cluster? By construction it should not.

Adversarial cases are reported SEPARATELY, never averaged into the headline. Expect
worse performance there; that is the honest ceiling of the current approach.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class CaseResult:
    case_id: str
    capability: str
    control: bool
    adversarial: str | None
    # conjunction
    set_recall: float | None = None
    found_conjunction_only: bool | None = None
    n_sets_found: int = 0
    pairwise_recall: float | None = None
    # missing hypothesis
    cluster_recall: float | None = None
    cluster_purity: float | None = None
    abduced: str = ""
    retrieval_recall: float | None = None
    n_hypotheses_proposed: int = 0
    # shared
    error: str = ""
    notes: list[str] = field(default_factory=list)


def load_cases(path: str | Path) -> list[dict]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _load_corpus(client, case: dict, log, keep_cache: bool = False):
    """Reset the system and load this case's data points + hypotheses over HTTP.
    Each data point carries meta.source = its case id, so results can be mapped back."""
    client.post("/api/reset", params={"in_memory": not keep_cache,
                                      "keep_cache": keep_cache})
    for dp in case["data_points"]:
        # atomize=False: authored cases are already single propositions. Re-atomizing
        # them splits one statement into fragments and destroys the planted structure.
        client.post("/api/episodes",
                    json_body={"content": dp["text"], "atomize": False,
                               "meta": {"source": dp["id"], "credibility": 1.0}})
    hyps = (case.get("given_hypotheses")
            or ([case["hypothesis"]] if case.get("hypothesis") else []))
    for h in hyps:
        client.post("/api/hypotheses", json_body={"text": h})
    return hyps


def _source_map(matrix: dict) -> dict:
    """system dp_id -> case data point id (from meta.source)."""
    return {dp["dp_id"]: dp.get("source") for dp in matrix.get("data_points", [])}


def run_conjunction_case(client, log, case: dict, keep_cache: bool = False) -> CaseResult:
    r = CaseResult(case_id=case["id"], capability=case["capability"],
                   control=bool(case.get("control")),
                   adversarial=case.get("adversarial"))
    log.step(case["id"], f"conjunction | {case.get('domain','?')} | "
                         f"control={r.control} adversarial={r.adversarial}")
    _load_corpus(client, case, log, keep_cache)
    m = client.post("/api/matrix")
    if m.status_code != 200:
        r.error = f"matrix failed {m.status_code}"
        return r
    matrix = m.json()
    smap = _source_map(matrix)
    if not matrix["hypotheses"]:
        r.error = "no hypotheses loaded"
        return r
    hyp_id = matrix["hypotheses"][0]["hyp_id"]

    resp = client.post(f"/api/hypotheses/{hyp_id}/refutations",
                       json_body={"max_size": 5, "n_seeds": 3, "shortlist": 20})
    if resp.status_code != 200:
        r.error = f"refutations failed {resp.status_code}"
        return r
    body = resp.json()
    # annotate with case ids so the dump is readable without cross-referencing hashes
    for st in body.get("refuting_sets", []):
        st["case_ids"] = [smap.get(i) for i in st["dp_ids"]]
    body["pairwise_case_ids"] = [smap.get(i)
                                 for i in body.get("pairwise_refuting_dp_ids", [])]
    body["planted"] = case["ground_truth"].get("refuting_sets")
    # Log the CASE-ID view: the raw response carries content hashes, which are
    # unreadable against the authored ground truth.
    log.detail(f"planted: {case['ground_truth'].get('refuting_sets')}")
    for st in body.get("refuting_sets", []):
        log.detail(f"found  : {st['case_ids']} joint_lift={st['joint_lift']} "
                   f"conjunction_only={st['conjunction_only']} "
                   f"member_lifts={ {smap.get(k): v for k, v in st['member_lifts'].items()} }")
    log.detail(f"pairwise baseline: {body['pairwise_case_ids']}")
    found_sets = [{smap.get(i) for i in s["dp_ids"]} for s in body["refuting_sets"]]
    r.n_sets_found = len(found_sets)
    planted = [set(s) for s in (case["ground_truth"].get("refuting_sets") or [])]

    if r.control:
        # any set reported here is a false positive
        r.set_recall = None
        r.notes.append(f"control: {r.n_sets_found} set(s) reported "
                       f"({'FALSE POSITIVE' if r.n_sets_found else 'clean'})")
    elif planted:
        hits = 0
        for p in planted:
            # a hit = the planted set was found exactly, or as a subset of a found set
            if any(p <= f for f in found_sets):
                hits += 1
        r.set_recall = hits / len(planted)
        if body["refuting_sets"]:
            r.found_conjunction_only = any(s["conjunction_only"]
                                           for s in body["refuting_sets"])

    # BASELINE: what pairwise-only refutation finds of the planted members
    pw = {smap.get(i) for i in body.get("pairwise_refuting_dp_ids", [])}
    if planted and not r.control:
        members = set().union(*planted)
        r.pairwise_recall = len(pw & members) / max(len(members), 1)

    log.detail(f"found {r.n_sets_found} set(s) | recall={r.set_recall} "
               f"| pairwise_baseline_recall={r.pairwise_recall}")
    return r


def run_missing_hypothesis_case(client, log, case: dict, keep_cache: bool = False) -> CaseResult:
    r = CaseResult(case_id=case["id"], capability=case["capability"],
                   control=bool(case.get("control")),
                   adversarial=case.get("adversarial"))
    log.step(case["id"], f"missing-hypothesis | {case.get('domain','?')} | "
                         f"control={r.control} adversarial={r.adversarial}")
    _load_corpus(client, case, log, keep_cache)
    m = client.post("/api/matrix")
    if m.status_code != 200:
        r.error = f"matrix failed {m.status_code}"
        return r
    smap = _source_map(m.json())

    resp = client.post("/api/residuals",
                       json_body={"epsilon": 0.15, "min_cluster_size": 3,
                                  "verify": True})
    if resp.status_code != 200:
        r.error = f"residuals failed {resp.status_code}"
        return r
    body = resp.json()
    for cl in body.get("clusters", []):
        cl["case_ids"] = [smap.get(i) for i in cl["dp_ids"]]
    body["planted_cluster"] = case["ground_truth"].get("cluster_members")
    body["planted_hypothesis"] = case["ground_truth"].get("missing_hypothesis")
    log.detail(f"planted cluster: {case['ground_truth'].get('cluster_members')}")
    for cl in body.get("clusters", []):
        log.detail(f"cluster: {cl['case_ids']} coherence={cl['coherence']} "
                   f"admitted={cl['admitted']} -> {cl.get('abduced_hypothesis','')[:70]}")
    clusters = body.get("clusters", [])
    admitted = [c for c in clusters if c.get("admitted")]
    r.n_hypotheses_proposed = len(admitted)

    planted = set(case["ground_truth"].get("cluster_members") or [])
    if r.control:
        r.notes.append(f"control: {len(admitted)} hypothesis(es) proposed "
                       f"({'FALSE HYPOTHESIS' if admitted else 'clean'})")
    elif planted:
        # best-matching cluster wins (the system may return several)
        best_recall, best_purity, best_abduced = 0.0, 0.0, ""
        for c in clusters:
            got = {smap.get(i) for i in c["dp_ids"]}
            rec = len(got & planted) / max(len(planted), 1)
            pur = len(got & planted) / max(len(got), 1)
            if rec > best_recall:
                best_recall, best_purity = rec, pur
                best_abduced = c.get("abduced_hypothesis", "")
        r.cluster_recall, r.cluster_purity, r.abduced = (best_recall, best_purity,
                                                         best_abduced)

    # BASELINE: does similarity ranking against the GIVEN hypotheses surface the cluster?
    if planted:
        retr = {smap.get(i) for i in body.get("retrieval_baseline_dp_ids", [])}
        r.retrieval_recall = len(retr & planted) / max(len(planted), 1)

    log.detail(f"residuals={body.get('n_residuals')}/{body.get('n_data_points')} "
               f"clusters={len(clusters)} admitted={len(admitted)} "
               f"| cluster_recall={r.cluster_recall} retrieval_baseline={r.retrieval_recall}")
    if r.abduced:
        log.detail(f"abduced: {r.abduced[:90]}")
    return r


_CAPABILITY_ALIASES = {
    "conjunction": "conjunction_refutation",
    "missing": "missing_hypothesis",
}


def run_capability_eval(client, log, cases_path: str, capability: str = "both",
                        only: str | None = None,
                        keep_cache: bool = False) -> list[CaseResult]:
    cases = load_cases(cases_path)
    if capability and capability != "both":
        want = _CAPABILITY_ALIASES.get(capability, capability)
        cases = [c for c in cases if c["capability"] == want]
    if only:
        cases = [c for c in cases if c["id"] == only]
    results = []
    for case in cases:
        try:
            if case["capability"] == "conjunction_refutation":
                results.append(run_conjunction_case(client, log, case, keep_cache))
            else:
                results.append(run_missing_hypothesis_case(client, log, case, keep_cache))
        except Exception as e:
            results.append(CaseResult(case_id=case["id"],
                                      capability=case["capability"],
                                      control=bool(case.get("control")),
                                      adversarial=case.get("adversarial"),
                                      error=f"unexpected: {e}"))
            log.detail(f"ERROR in {case['id']}: {e}")
    print_capability_report(results, log)
    return results


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _f(x):
    return "  —  " if x is None else f"{x:5.2f}"


def print_capability_report(results: list[CaseResult], log):
    log.always("")
    log.always("=" * 86)
    log.always("  CAPABILITY EVALUATION")
    log.always("=" * 86)

    conj = [r for r in results if r.capability == "conjunction_refutation"]
    miss = [r for r in results if r.capability == "missing_hypothesis"]

    log.always("\n(a) CONJUNCTION REFUTATION")
    log.always(f"  {'case':14s}{'recall':>8s}{'pairwise':>10s}{'sets':>6s}  note")
    for r in conj:
        note = ("CONTROL " if r.control else "") + (r.adversarial or "") + \
               (" ERROR:" + r.error if r.error else "")
        log.always(f"  {r.case_id:14s}{_f(r.set_recall):>8s}"
                   f"{_f(r.pairwise_recall):>10s}{r.n_sets_found:>6d}  {note}")
    main_c = [r for r in conj if not r.control and not r.adversarial]
    adv_c = [r for r in conj if not r.control and r.adversarial]
    ctrl_c = [r for r in conj if r.control]
    log.always(f"  MAIN     recall={_f(_mean([r.set_recall for r in main_c]))}"
               f"  pairwise_baseline={_f(_mean([r.pairwise_recall for r in main_c]))}")
    log.always(f"  ADVERSARIAL recall={_f(_mean([r.set_recall for r in adv_c]))}"
               f"   (reported separately — expected to be worse)")
    fp = sum(r.n_sets_found for r in ctrl_c)
    log.always(f"  CONTROLS false positives: {fp} set(s) across {len(ctrl_c)} control case(s)")

    log.always("\n(b) MISSING-HYPOTHESIS DETECTION")
    log.always(f"  {'case':14s}{'recall':>8s}{'purity':>8s}{'retrieval':>11s}  abduced / note")
    for r in miss:
        note = ("CONTROL " if r.control else "") + (r.adversarial or "")
        if r.error:
            note += " ERROR:" + r.error
        txt = (r.abduced[:44] + "…") if len(r.abduced) > 45 else r.abduced
        log.always(f"  {r.case_id:14s}{_f(r.cluster_recall):>8s}{_f(r.cluster_purity):>8s}"
                   f"{_f(r.retrieval_recall):>11s}  {txt or note}")
    main_m = [r for r in miss if not r.control and not r.adversarial]
    adv_m = [r for r in miss if not r.control and r.adversarial]
    ctrl_m = [r for r in miss if r.control]
    log.always(f"  MAIN     cluster_recall={_f(_mean([r.cluster_recall for r in main_m]))}"
               f"  retrieval_baseline={_f(_mean([r.retrieval_recall for r in main_m]))}")
    log.always(f"  ADVERSARIAL cluster_recall={_f(_mean([r.cluster_recall for r in adv_m]))}")
    fh = sum(r.n_hypotheses_proposed for r in ctrl_m)
    log.always(f"  CONTROLS false hypotheses: {fh} across {len(ctrl_m)} control case(s)")

    log.always("")
    log.always("READ THIS AS: the capability's value is the GAP between the system column")
    log.always("and its baseline column. A high recall that the baseline also achieves is")
    log.always("not differentiated value. Abduction text is for human judging, not scored.")
    log.always("")
