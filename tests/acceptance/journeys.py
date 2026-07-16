"""Acceptance scenarios — the actual end-to-end checks, as ordered steps.

Each scenario is a function taking (client, log) and raising AssertionError on
failure (via log.check). They exercise real user journeys through the HTTP API and
work identically against a local server or a cloud deployment.

These hit the REAL system, including whatever LLM the deployment is configured with.
They therefore assert BEHAVIOURAL properties that hold regardless of the specific
model (structure, citations, ordering, sanity) — never exact scores, which are
model-dependent.
"""
from __future__ import annotations
from .http_client import ApiClient, StepLogger


def journey_reset_and_empty(client: ApiClient, log: StepLogger):
    log.banner("Scenario: reset yields an empty system")
    log.step("1", "Reset the system to a clean in-memory state")
    r = client.post("/api/reset", params={"in_memory": True})
    log.check(r.status_code == 200, "reset returns 200", r.status_code)

    log.step("2", "Matrix on an empty corpus is empty")
    r = client.post("/api/matrix")
    log.check(r.status_code == 200, "matrix returns 200", r.status_code)
    body = r.json()
    log.check(body["hypotheses"] == [], "no hypotheses", body["hypotheses"])
    log.check(body["data_points"] == [], "no data points", body["data_points"])


def journey_ingest_and_atomize(client: ApiClient, log: StepLogger):
    log.banner("Scenario: ingest a document, atomize into data points")
    client.post("/api/reset", params={"in_memory": True})

    log.step("1", "POST a multi-sentence document")
    text = ("The oven temperature sensors reported stable readings all night. "
            "No fault codes were raised by the controller. "
            "The bakery switched to a new flour supplier in November. "
            "Deliveries from that supplier arrived late three times.")
    r = client.post("/api/documents",
                    json_body={"name": "incident", "text": text,
                               "meta": {"source": "incident_report"}})
    log.check(r.status_code == 200, "document accepted", r.status_code)
    body = r.json()
    log.detail(f"atomized into {body['n_data_points']} data points")
    for dp in body.get("data_points", []):
        log.detail(f"  • {dp}")
    log.check(body["n_data_points"] >= 2, "document atomized into >= 2 data points",
              body["n_data_points"])
    log.check(any("flour" in dp.lower() for dp in body["data_points"]),
              "a flour-related data point is present")


def journey_full_matrix(client: ApiClient, log: StepLogger):
    log.banner("Scenario: load a scenario, build the full validation matrix")
    log.step("1", "Reset + load bundled scenario 0 (burnt bread)")
    client.post("/api/reset", params={"in_memory": True})
    r = client.post("/api/load_scenario", json_body={"index": 0})
    log.check(r.status_code == 200, "scenario loaded", r.status_code)
    loaded = r.json()
    log.detail(f"loaded '{loaded['loaded']}': "
               f"{loaded['n_episodes']} episodes, {loaded['n_hypotheses']} hypotheses")

    log.step("2", "Build the matrix")
    r = client.post("/api/matrix")
    log.check(r.status_code == 200, "matrix built", r.status_code)
    m = r.json()
    n_h, n_d = len(m["hypotheses"]), len(m["data_points"])
    log.detail(f"matrix: {n_h} hypotheses x {n_d} data points = {n_h*n_d} cells")

    log.step("3", "Every hypothesis-datapoint pair has a cell")
    log.check(len(m["cells"]) == n_h * n_d, "full grid present",
              f"{len(m['cells'])} != {n_h*n_d}")

    log.step("4", "Hypotheses are returned ranked by score (descending)")
    scores = [h["score"] for h in m["hypotheses"]]
    log.detail("scores: " + ", ".join(f"{s:+.2f}" for s in scores))
    log.check(scores == sorted(scores, reverse=True), "scores descending", scores)

    log.step("5", "Every cell has a finite log-lift and a verdict")
    import math
    bad = [k for k, c in m["cells"].items()
           if not math.isfinite(c["log_lift"]) or not c.get("verdict")]
    log.check(not bad, "all cells finite and verdicted", bad[:5])


def journey_hypothesis_detail_cited(client: ApiClient, log: StepLogger):
    log.banner("Scenario: hypothesis detail is cited and ordered")
    client.post("/api/reset", params={"in_memory": True})
    client.post("/api/load_scenario", json_body={"index": 3})  # insider trading
    m = client.post("/api/matrix").json()
    hid = m["hypotheses"][0]["hyp_id"]

    log.step("1", f"Fetch detail for top hypothesis {hid}")
    r = client.get(f"/api/hypotheses/{hid}")
    log.check(r.status_code == 200, "detail returns 200", r.status_code)
    d = r.json()
    log.detail(f"hypothesis: {d['hypothesis']}")
    log.detail(f"verdict: {d['verdict']}  score: {d['score']:+.2f}")

    log.step("2", "Evidence is present and each item is cited")
    log.check(bool(d["evidence"]), "evidence present")
    for e in d["evidence"][:5]:
        log.detail(f"  [{e['verdict']:20s}] lift={e['lift']:.2f} "
                   f"src={e.get('source')}  {e['data_point'][:60]}")
    log.check(all(e["data_point"] and "source" in e for e in d["evidence"]),
              "every evidence item has text and a source citation")

    log.step("3", "Evidence ordered most-contradicting first")
    lifts = [e["log_lift"] for e in d["evidence"]]
    log.check(lifts == sorted(lifts), "log-lift ascending (contradicting first)", lifts)


def journey_unknown_hypothesis_404(client: ApiClient, log: StepLogger):
    log.banner("Scenario: unknown hypothesis id returns 404")
    client.post("/api/reset", params={"in_memory": True})
    client.post("/api/load_scenario", json_body={"index": 0})
    log.step("1", "GET a non-existent hypothesis id")
    r = client.get("/api/hypotheses/deadbeef")
    log.check(r.status_code == 404, "returns 404", r.status_code)


def journey_noise_robustness(client: ApiClient, log: StepLogger):
    log.banner("Scenario: injected noise stays low-diagnosticity and doesn't flip ranking")
    log.step("1", "Load scenario 0 clean, record top hypothesis")
    client.post("/api/reset", params={"in_memory": True})
    client.post("/api/load_scenario", json_body={"index": 0})
    clean = client.post("/api/matrix").json()
    clean_top = clean["hypotheses"][0]["text"]
    log.detail(f"clean top hypothesis: {clean_top[:70]}")

    log.step("2", "Load scenario 0 with noise + duplicates")
    client.post("/api/reset", params={"in_memory": True})
    client.post("/api/load_scenario",
                json_body={"index": 0, "noise": 6, "duplicates": 2})
    noisy = client.post("/api/matrix").json()
    noisy_top = noisy["hypotheses"][0]["text"]
    log.detail(f"noisy top hypothesis: {noisy_top[:70]}")

    log.step("3", "Top hypothesis unchanged by noise")
    log.check(clean_top == noisy_top, "noise did not flip the ranking",
              f"{clean_top!r} != {noisy_top!r}")

    log.step("4", "Injected noise data points are less diagnostic than real evidence")
    noise_kw = ("cafeteria", "parking", "espresso", "softball",
                "recycling", "password", "artwork")
    noise_d = [dp["diagnosticity"] for dp in noisy["data_points"]
               if any(k in dp["text"].lower() for k in noise_kw)]
    real_d = [dp["diagnosticity"] for dp in noisy["data_points"]]
    if noise_d:
        mn, mr = sum(noise_d)/len(noise_d), sum(real_d)/len(real_d)
        log.detail(f"mean diagnosticity — noise {mn:.3f} vs all {mr:.3f}")
        log.check(mn <= mr + 1e-9, "noise no more diagnostic than real evidence",
                  f"{mn:.3f} > {mr:.3f}")
    else:
        log.note("no noise data points detected (atomization may have reworded them)")


# ordered list the runner executes
ALL_JOURNEYS = [
    journey_reset_and_empty,
    journey_ingest_and_atomize,
    journey_full_matrix,
    journey_hypothesis_detail_cited,
    journey_unknown_hypothesis_404,
    journey_noise_robustness,
]
