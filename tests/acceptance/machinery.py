"""Machinery checks — does the system WORK? (the pass/fail verdict layer)

These verify plumbing only: that ingest → chunk → atomize → score → matrix → cited
detail connects and returns well-formed data over HTTP. They are NOT about answer
quality; that is what --eval and --cases-file measure.

DESIGN NOTE — why the corpora here are tiny:
An earlier version loaded the bundled scenarios (103 data points x 4 hypotheses =
~412 cells) for four of these checks. Against a real model that cost ~40 minutes per
run — 91% of total runtime — to verify plumbing that an 8-cell corpus proves just as
well. Plumbing correctness does not depend on corpus size, so these use small
purpose-built corpora and run in seconds. That is what lets the verdict layer run
ALWAYS, on every invocation, without anyone wanting to skip it.

The one thing size DOES affect is semantic chunking of long documents, so
check_ingest_and_atomize still posts a genuine multi-sentence document.
"""
from __future__ import annotations
from .http_client import ApiClient, StepLogger

# A compact corpus with clear structure: two hypotheses, a few data points that
# discriminate between them, plus one obviously irrelevant line used as noise.
_CORPUS = [
    ("d1", "The bakery switched to a new flour supplier in November."),
    ("d2", "Only loaves baked with the new supplier's flour came out overbaked."),
    ("d3", "The oven thermostats were inspected on Monday and found accurate."),
    ("d4", "The office cafeteria introduced a new seasonal pumpkin soup."),   # noise
]
_HYPOTHESES = [
    "The new supplier's flour is defective.",
    "The oven thermostats are malfunctioning.",
]


def _load_small_corpus(client: ApiClient, log: StepLogger, keep_cache: bool = False):
    client.post("/api/reset", params={"in_memory": not keep_cache,
                                      "keep_cache": keep_cache})
    for src, text in _CORPUS:
        r = client.post("/api/episodes",
                        json_body={"content": text,
                                   "meta": {"source": src, "credibility": 1.0,
                                            "noise": "irrelevant" if src == "d4" else None}})
        # Assert the load: silently ignoring a failed post produced an empty corpus
        # and misleading downstream failures that looked like scoring problems.
        log.check(r.status_code == 200, f"episode '{src}' loaded", r.status_code)
    for h in _HYPOTHESES:
        client.post("/api/hypotheses", json_body={"text": h})


def check_reset_yields_empty_system(client: ApiClient, log: StepLogger, keep_cache=False):
    log.banner("Check: reset yields an empty system")
    log.step("1", "Reset to a clean state")
    r = client.post("/api/reset", params={"in_memory": True})
    log.check(r.status_code == 200, "reset returns 200", r.status_code)

    log.step("2", "Matrix on an empty corpus is empty")
    r = client.post("/api/matrix")
    log.check(r.status_code == 200, "matrix returns 200", r.status_code)
    b = r.json()
    log.check(b["hypotheses"] == [], "no hypotheses", b["hypotheses"])
    log.check(b["data_points"] == [], "no data points", b["data_points"])


def check_ingest_and_atomize(client: ApiClient, log: StepLogger, keep_cache=False):
    log.banner("Check: a document is chunked and atomized into data points")
    client.post("/api/reset", params={"in_memory": not keep_cache,
                                      "keep_cache": keep_cache})
    log.step("1", "POST a multi-sentence document (exercises semantic chunking)")
    text = ("The oven temperature sensors reported stable readings all night. "
            "No fault codes were raised by the controller. "
            "The bakery switched to a new flour supplier in November. "
            "Deliveries from that supplier arrived late three times.")
    r = client.post("/api/documents",
                    json_body={"name": "incident", "text": text,
                               "meta": {"source": "incident_report"}})
    log.check(r.status_code == 200, "document accepted", r.status_code)
    b = r.json()
    for dp in b.get("data_points", []):
        log.detail(f"  • {dp}")
    log.check(b["n_data_points"] >= 2, "atomized into >= 2 data points",
              b["n_data_points"])
    # Upper bound matters as much as the lower one. Over-splitting shreds a fact into
    # fragments that are individually uninformative ("An inspection was conducted"),
    # which dilutes diagnosticity, inflates cost, and pollutes residual clustering.
    # The document has 4 sentences; more than ~2 data points per sentence means the
    # atomizer is decomposing below the level of a fact.
    n_sentences = 4
    log.check(b["n_data_points"] <= 2 * n_sentences,
              f"not over-split (<= {2*n_sentences} data points from {n_sentences} sentences)",
              b["n_data_points"])
    log.check(any("flour" in dp.lower() for dp in b["data_points"]),
              "a flour-related data point is present")


def check_full_matrix(client: ApiClient, log: StepLogger, keep_cache=False):
    log.banner("Check: the full validation matrix is well-formed")
    log.step("1", "Load a small corpus + 2 hypotheses")
    _load_small_corpus(client, log, keep_cache)

    log.step("2", "Build the matrix")
    r = client.post("/api/matrix")
    log.check(r.status_code == 200, "matrix built", r.status_code)
    m = r.json()
    n_h, n_d = len(m["hypotheses"]), len(m["data_points"])
    log.detail(f"{n_h} hypotheses x {n_d} data points = {n_h*n_d} cells")

    log.step("3", "Every hypothesis-datapoint pair has a cell")
    log.check(len(m["cells"]) == n_h * n_d, "full grid present",
              f"{len(m['cells'])} != {n_h*n_d}")

    log.step("4", "Hypotheses are ranked by score (descending)")
    scores = [h["score"] for h in m["hypotheses"]]
    log.detail("scores: " + ", ".join(f"{s:+.2f}" for s in scores))
    log.check(scores == sorted(scores, reverse=True), "scores descending", scores)

    log.step("5", "Every cell has a finite log-lift and a verdict")
    import math
    bad = [k for k, c in m["cells"].items()
           if not math.isfinite(c["log_lift"]) or not c.get("verdict")]
    log.check(not bad, "all cells finite and verdicted", bad[:5])


def check_hypothesis_detail_is_cited(client: ApiClient, log: StepLogger, keep_cache=False):
    log.banner("Check: hypothesis detail is cited and ordered")
    _load_small_corpus(client, log, keep_cache)
    m = client.post("/api/matrix").json()
    hid = m["hypotheses"][0]["hyp_id"]

    log.step("1", f"Fetch detail for top hypothesis {hid}")
    r = client.get(f"/api/hypotheses/{hid}")
    log.check(r.status_code == 200, "detail returns 200", r.status_code)
    d = r.json()
    log.detail(f"hypothesis: {d['hypothesis']}")

    log.step("2", "Evidence present and every item carries a citation")
    log.check(bool(d["evidence"]), "evidence present")
    for e in d["evidence"]:
        log.detail(f"  [{e['verdict']:20s}] lift={e['lift']:.2f} src={e.get('source')}")
    log.check(all(e["data_point"] and "source" in e for e in d["evidence"]),
              "every evidence item has text and a source citation")

    log.step("3", "Evidence ordered most-contradicting first")
    lifts = [e["log_lift"] for e in d["evidence"]]
    log.check(lifts == sorted(lifts), "log-lift ascending", lifts)


def check_unknown_hypothesis_404(client: ApiClient, log: StepLogger, keep_cache=False):
    log.banner("Check: unknown hypothesis id returns 404")
    _load_small_corpus(client, log, keep_cache)
    log.step("1", "GET a non-existent hypothesis id")
    r = client.get("/api/hypotheses/deadbeef")
    log.check(r.status_code == 404, "returns 404", r.status_code)


def check_noise_stays_low_diagnosticity(client: ApiClient, log: StepLogger, keep_cache=False):
    log.banner("Check: irrelevant data stays low-diagnosticity")
    _load_small_corpus(client, log, keep_cache)
    m = client.post("/api/matrix").json()
    log.step("1", "The off-topic data point is no more diagnostic than the rest")
    noise = [dp["diagnosticity"] for dp in m["data_points"]
             if "cafeteria" in dp["text"].lower() or "pumpkin" in dp["text"].lower()]
    allq = [dp["diagnosticity"] for dp in m["data_points"]]
    if not noise:
        log.note("noise data point not found (atomization may have reworded it)")
        return
    mn, ma = sum(noise)/len(noise), sum(allq)/len(allq)
    log.detail(f"mean diagnosticity — noise {mn:.3f} vs all {ma:.3f}")
    log.check(mn <= ma + 1e-9, "noise no more diagnostic than average", f"{mn} > {ma}")


ALL_CHECKS = [
    check_reset_yields_empty_system,
    check_ingest_and_atomize,
    check_full_matrix,
    check_hypothesis_detail_is_cited,
    check_unknown_hypothesis_404,
    check_noise_stays_low_diagnosticity,
]
