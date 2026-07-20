"""FastAPI API + FastHTML UI for step 1.

User stories exposed (step-1 subset):
  US-INGEST   POST /api/documents         add a document (chunked + atomized)
  US-EPISODE  POST /api/episodes          add a pre-atomized episode
  US-HYP      POST /api/hypotheses        add a hypothesis
  US-MATRIX   POST /api/matrix            score everything, return the matrix
  US-DETAIL   GET  /api/hypotheses/{id}   scored evidence for one hypothesis (cited)
  US-LOAD     POST /api/load_scenario     load a bundled YAML scenario (+optional noise)

UI:
  GET /            matrix view (hypotheses x data points, coloured by lift)
  GET /hyp/{id}    hypothesis detail with cited, ranked evidence

Run:  uvicorn counterpoint.app:app --reload
The app uses a single in-process Pipeline (SQLite-backed). Mock LLM if no API key.
"""
from __future__ import annotations
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .config import Settings
from .pipeline import Pipeline
from .fixtures import all_scenarios, inject_noise, load_into_pipeline
from .models import Matrix

settings = Settings()
pipeline = Pipeline(settings)
app = FastAPI(title="Counterpoint (PoC step 1)")


# --------------------------------------------------------------- API schemas
class DocIn(BaseModel):
    name: str
    text: str
    meta: dict = {}


class EpisodeIn(BaseModel):
    content: str
    meta: dict = {}
    atomize: bool = True   # False when the caller already supplies atomic statements


class HypIn(BaseModel):
    text: str
    context_summary: str = ""


class LoadIn(BaseModel):
    index: int = 0
    noise: int = 0
    duplicates: int = 0


def _new_pipeline(in_memory: bool = False) -> Pipeline:
    s = settings
    if in_memory:
        from copy import copy
        s = copy(settings)
        s.cache_db_path = ":memory:"
    from .store import Store
    return Pipeline(s, store=Store(s.cache_db_path))


@app.post("/api/reset")
def reset(in_memory: bool = True, keep_cache: bool = False):
    """Clear corpus and hypotheses so the system can be driven from an empty state.

    keep_cache=true clears only the corpus and PRESERVES the lift/prior caches, so a
    re-run of the same texts costs no LLM calls. Use this with a file-backed
    COUNTERPOINT_CACHE_DB_PATH to make repeated evaluation runs nearly free.
    in_memory=true (the default) swaps in a throwaway DB, discarding everything."""
    global pipeline
    if keep_cache:
        pipeline.store.clear_corpus()
        return {"status": "reset", "in_memory": False, "kept_cache": True}
    pipeline = _new_pipeline(in_memory=in_memory)
    if not in_memory:
        # A fresh Store object does NOT empty an existing FILE database — it just
        # reopens it. Without this, a file-backed reset silently kept every previous
        # corpus, so cases accumulated into one another.
        pipeline.store.clear_all()
    return {"status": "reset", "in_memory": in_memory, "kept_cache": False}


# --------------------------------------------------------------- API routes
@app.post("/api/documents")
def add_document(d: DocIn):
    doc = pipeline.add_document(d.name, d.text, d.meta)
    dps = [dp for dp in pipeline.store.data_points() if dp.doc_id == doc.doc_id]
    return {"doc_id": doc.doc_id, "n_data_points": len(dps),
            "data_points": [dp.text for dp in dps]}


@app.post("/api/episodes")
def add_episode(inp: EpisodeIn):
    """Add one episode. By default the text is atomized into data points.

    atomize=false stores the text AS a single data point. Use this when the caller
    already supplies atomic statements (e.g. authored evaluation cases): re-atomizing
    them shreds single propositions into fragments like "A physical inspection was
    conducted.", which carry no information and pollute clustering.
    """
    if inp.atomize:
        doc = pipeline.add_episode(inp.content, meta=inp.meta)
        dps = [dp for dp in pipeline.store.data_points() if dp.doc_id == doc.doc_id]
        return {"doc_id": doc.doc_id, "n_data_points": len(dps),
                "data_points": [dp.text for dp in dps]}
    res = pipeline.add_atomic(inp.content, meta=inp.meta)
    return {"doc_id": res["doc_id"], "n_data_points": 1,
            "data_points": [inp.content]}
@app.post("/api/hypotheses")
def add_hypothesis(h: HypIn):
    hyp = pipeline.add_hypothesis(h.text, h.context_summary)
    return {"hyp_id": hyp.hyp_id, "text": hyp.text}


@app.post("/api/load_scenario")
def load_scenario(inp: LoadIn):
    scs = all_scenarios()
    if not (0 <= inp.index < len(scs)):
        raise HTTPException(404, f"scenario index out of range (0..{len(scs)-1})")
    sc = scs[inp.index]
    if inp.noise or inp.duplicates:
        sc = inject_noise(sc, n=inp.noise, duplicate=inp.duplicates)
    load_into_pipeline(pipeline, sc)
    return {"loaded": sc.name, "n_episodes": len(sc.episodes),
            "n_hypotheses": len(sc.hypotheses)}


@app.post("/api/matrix")
def get_matrix():
    m = pipeline.matrix()
    return _matrix_json(m)


@app.get("/api/hypotheses/{hyp_id}")
def hypothesis_detail(hyp_id: str):
    m = pipeline.matrix()
    res = next((r for r in m.results if r.hypothesis.hyp_id == hyp_id), None)
    if not res:
        raise HTTPException(404, "hypothesis not found")
    dp_by = {dp.dp_id: dp for dp in m.data_points}
    ev = sorted(res.cells, key=lambda c: c.log_lift)
    return {
        "hypothesis": res.hypothesis.text,
        "score": res.score, "verdict": res.verdict,
        "counts": {"supporting": res.n_supporting,
                   "contradicting": res.n_contradicting, "neutral": res.n_neutral},
        "evidence": [{
            "data_point": dp_by[c.dp_id].text,
            "source": dp_by[c.dp_id].meta.get("source") or dp_by[c.dp_id].doc_id,
            "credibility": dp_by[c.dp_id].meta.get("credibility"),
            "lift": round(c.lift, 3), "log_lift": round(c.log_lift, 3),
            "verdict": c.verdict, "relatedness": round(c.relatedness, 3),
            "diagnosticity": round(m.diagnosticity.get(c.dp_id, 0.0), 3),
            "rationale": c.rationale,
        } for c in ev],
    }


class RefuteIn(BaseModel):
    max_size: int = 4
    n_seeds: int = 3
    shortlist: int = 20


class ResidualIn(BaseModel):
    epsilon: float = 0.15
    min_cluster_size: int = 3
    # Cosine distance at which agglomerative clustering stops merging. Lower = tighter,
    # more clusters. Clustering makes NO LLM calls, so this can be swept for free
    # (set verify=false to skip abduction entirely while tuning).
    distance_threshold: float = 0.6
    verify: bool = True


@app.post("/api/hypotheses/{hyp_id}/refutations")
def find_refutations(hyp_id: str, inp: RefuteIn):
    """Capability (a): minimal sets of data points that JOINTLY refute this hypothesis
    while (ideally) no member refutes it alone."""
    from .conjunction import find_refuting_sets, pairwise_refutations
    m = pipeline.matrix()
    h = next((r.hypothesis for r in m.results if r.hypothesis.hyp_id == hyp_id), None)
    if not h:
        raise HTTPException(404, "hypothesis not found")
    dp_by = {dp.dp_id: dp for dp in m.data_points}
    sets = find_refuting_sets(h, m.data_points, pipeline.scorer,
                              max_size=inp.max_size, n_seeds=inp.n_seeds,
                              shortlist=inp.shortlist)
    return {
        "hypothesis": h.text,
        "refuting_sets": [{
            "dp_ids": s.dp_ids,
            "data_points": [dp_by[i].text for i in s.dp_ids if i in dp_by],
            "size": s.size,
            "joint_lift": round(s.joint_lift, 4),
            "member_lifts": {k: round(v, 4) for k, v in s.member_lifts.items()},
            "conjunction_only": s.is_conjunction_only,
            "rationale": s.rationale,
            "llm_calls": s.llm_calls,
        } for s in sets],
        # baseline: what a pairwise checker would find on its own
        "pairwise_refuting_dp_ids": pairwise_refutations(h, m.data_points,
                                                         pipeline.scorer),
    }


@app.post("/api/residuals")
def detect_residuals(inp: ResidualIn):
    """Capability (b): coherent clusters of data points no hypothesis explains, each
    with an abduced missing hypothesis."""
    from .residual import (detect_missing_hypotheses, find_residuals,
                           retrieval_baseline)
    m = pipeline.matrix()
    dp_by = {dp.dp_id: dp for dp in m.data_points}
    residual_dps = find_residuals(m.hypotheses, m.data_points, m.cells, inp.epsilon)
    clusters = detect_missing_hypotheses(
        m.hypotheses, m.data_points, m.cells, pipeline.scorer, pipeline.backend,
        epsilon=inp.epsilon, min_cluster_size=inp.min_cluster_size,
        distance_threshold=inp.distance_threshold, verify=inp.verify)
    return {
        "n_data_points": len(m.data_points),
        "n_residuals": len(residual_dps),
        "residual_dp_ids": [dp.dp_id for dp in residual_dps],
        "clusters": [{
            "dp_ids": c.dp_ids,
            "data_points": [dp_by[i].text for i in c.dp_ids if i in dp_by],
            "size": c.size,
            "coherence": round(c.coherence, 4),
            "abduced_hypothesis": c.abduced_hypothesis,
            "why": c.abduction_why,
            "explains_fraction": round(c.explains_fraction, 3),
            "admitted": c.admitted,
            "notes": c.notes,
        } for c in clusters],
        # baseline: what similarity-ranked retrieval would surface instead
        "retrieval_baseline_dp_ids": retrieval_baseline(m.hypotheses, m.data_points,
                                                        pipeline.backend),
    }


def _llm_stats() -> dict:
    """Successes/failures of the underlying LLM client, if it is the real one.

    A failed scoring call degrades to caller defaults (lift 1.0 = "no bearing"),
    which is indistinguishable from a genuine neutral verdict. Surfacing the counts
    means silent degradation can be noticed rather than mistaken for a result.
    """
    llm = getattr(getattr(pipeline, "scorer", None), "llm", None) \
          or getattr(pipeline, "llm", None)
    ok = getattr(llm, "_successes", None)
    bad = getattr(llm, "_failures", None)
    if ok is None:
        return {}
    return {"llm_calls_ok": ok, "llm_calls_failed": bad,
            "degraded": bool(bad),
            # the actual failure messages — without these, a non-zero failure count
            # says something is wrong but not what
            "errors": list(getattr(llm, "_recent_errors", []))[:8]}


def _matrix_json(m: Matrix) -> dict:
    dp_by = {dp.dp_id: dp for dp in m.data_points}
    return {
        "hypotheses": [{"hyp_id": r.hypothesis.hyp_id, "text": r.hypothesis.text,
                        "score": round(r.score, 3), "verdict": r.verdict,
                        "supporting": r.n_supporting, "contradicting": r.n_contradicting,
                        "neutral": r.n_neutral} for r in m.results],
        "data_points": [{"dp_id": dp.dp_id, "text": dp.text,
                         "source": dp.meta.get("source") or dp.doc_id,
                         "credibility": dp.meta.get("credibility"),
                         "noise": dp.meta.get("noise"),
                         "diagnosticity": round(m.diagnosticity.get(dp.dp_id, 0.0), 3)}
                        for dp in m.data_points],
        "llm": _llm_stats(),
        "cells": {f"{h}|{d}": {"lift": round(c.lift, 3), "log_lift": round(c.log_lift, 3),
                               "verdict": c.verdict}
                  for (h, d), c in m.cells.items()},
    }


# --------------------------------------------------------------- FastHTML UI
try:
    from fasthtml.common import (Div, H1, H2, H3, P, A, Table, Tr, Td, Th, Span,
                                 Titled, Style, Form, Button, NotStr)
    _FASTHTML = True
except Exception:                        # pragma: no cover
    _FASTHTML = False

_VERDICT_COLOR = {
    "strongly_supports": "#0f6e56", "supports": "#1d9e75",
    "neutral": "#b4b2a9",
    "contradicts": "#d85a30", "strongly_contradicts": "#993c1d",
}
_CSS = """
body{font-family:system-ui,sans-serif;margin:2rem;color:#2c2c2a;line-height:1.5}
table{border-collapse:collapse;font-size:13px}
td,th{border:1px solid #d3d1c7;padding:6px 8px;vertical-align:top}
th{background:#f1efe8;text-align:left;font-weight:500}
.cell{width:34px;height:34px;text-align:center;color:#fff;font-weight:500;border-radius:4px}
.dp{max-width:520px}
.score{font-variant-numeric:tabular-nums}
a{color:#185fa5;text-decoration:none} a:hover{text-decoration:underline}
.pill{display:inline-block;padding:2px 8px;border-radius:10px;color:#fff;font-size:12px}
.muted{color:#5f5e5a;font-size:12px}
"""


def _cell_html(cell):
    color = _VERDICT_COLOR.get(cell.verdict, "#b4b2a9")
    tip = f"{cell.verdict} · lift {cell.lift:.2f} · {cell.rationale[:80]}"
    return Td(Div(f"{cell.log_lift:+.1f}", cls="cell", style=f"background:{color}",
                  title=tip))


@app.get("/", response_class=HTMLResponse)
def ui_matrix():
    if not _FASTHTML:
        return HTMLResponse("<p>FastHTML not installed.</p>")
    m = pipeline.matrix()
    if not m.hypotheses or not m.data_points:
        return HTMLResponse(_page(Div(
            H1("Counterpoint"),
            P("No corpus loaded. POST to /api/load_scenario or /api/documents, "
              "then reload."),
            P(A("API docs", href="/docs")))))
    dp_by = {dp.dp_id: dp for dp in m.data_points}
    dps_sorted = sorted(m.data_points,
                        key=lambda dp: -m.diagnosticity.get(dp.dp_id, 0.0))
    header = Tr(Th("data point ↓ / hypothesis →  (sorted by diagnosticity)"),
                *[Th(A(f"H{i+1}", href=f"/hyp/{r.hypothesis.hyp_id}",
                       title=r.hypothesis.text)) for i, r in enumerate(m.results)])
    rows = [header]
    for dp in dps_sorted:
        diag = m.diagnosticity.get(dp.dp_id, 0.0)
        label = Td(Div(dp.text, cls="dp"),
                   Div(f"{dp.meta.get('source','?')} · cred "
                       f"{dp.meta.get('credibility','—')} · diag {diag:.2f}", cls="muted"))
        cells = [_cell_html(m.cells[(r.hypothesis.hyp_id, dp.dp_id)])
                 for r in m.results]
        rows.append(Tr(label, *cells))
    legend = Div(*[Span(v.replace("_", " "), cls="pill",
                        style=f"background:{c};margin-right:6px")
                   for v, c in _VERDICT_COLOR.items()])
    summary = Table(Tr(Th("hypothesis"), Th("score"), Th("verdict"),
                       Th("sup"), Th("con"), Th("neu")),
                    *[Tr(Td(A(r.hypothesis.text, href=f"/hyp/{r.hypothesis.hyp_id}")),
                         Td(f"{r.score:+.2f}", cls="score"), Td(r.verdict),
                         Td(str(r.n_supporting)), Td(str(r.n_contradicting)),
                         Td(str(r.n_neutral))) for r in m.results])
    return HTMLResponse(_page(Div(
        H1("Validation matrix"),
        P("Cells show log-lift. Green = supports, grey = neutral/inert, "
          "orange-red = contradicts. Rows sorted by diagnosticity (top = most "
          "discriminating).", cls="muted"),
        legend, H2("Hypotheses, ranked by score"), summary,
        H2("Matrix"), Table(*rows))))


@app.get("/hyp/{hyp_id}", response_class=HTMLResponse)
def ui_hypothesis(hyp_id: str):
    if not _FASTHTML:
        return HTMLResponse("<p>FastHTML not installed.</p>")
    m = pipeline.matrix()
    res = next((r for r in m.results if r.hypothesis.hyp_id == hyp_id), None)
    if not res:
        return HTMLResponse("<p>not found</p>", status_code=404)
    dp_by = {dp.dp_id: dp for dp in m.data_points}
    ev = sorted(res.cells, key=lambda c: c.log_lift)
    rows = [Tr(Th("evidence (most contradicting first)"), Th("verdict"),
               Th("lift"), Th("rel"), Th("diag"), Th("source"), Th("cred"))]
    for c in ev:
        dp = dp_by[c.dp_id]
        rows.append(Tr(
            Td(Div(dp.text, cls="dp"), Div(c.rationale, cls="muted")),
            Td(Span(c.verdict.replace("_", " "), cls="pill",
                    style=f"background:{_VERDICT_COLOR.get(c.verdict,'#b4b2a9')}")),
            Td(f"{c.lift:.2f}", cls="score"), Td(f"{c.relatedness:.2f}"),
            Td(f"{m.diagnosticity.get(c.dp_id,0):.2f}"),
            Td(dp.meta.get("source", dp.doc_id)),
            Td(str(dp.meta.get("credibility", "—")))))
    return HTMLResponse(_page(Div(
        A("← matrix", href="/"),
        H1("Hypothesis"), P(res.hypothesis.text),
        P(f"score {res.score:+.2f} · verdict {res.verdict} · "
          f"{res.n_supporting} supporting / {res.n_contradicting} contradicting / "
          f"{res.n_neutral} neutral", cls="muted"),
        Table(*rows))))


def _page(body) -> str:
    from fasthtml.common import to_xml, Html, Head, Body, Title
    return to_xml(Html(Head(Title("Counterpoint"), Style(_CSS)), Body(body)))


# ---------------------------------------------------------------- error visibility
@app.exception_handler(Exception)
async def _unhandled(request, exc):
    """Return the actual error and traceback instead of a bare 500.

    This is a PoC/diagnostic service: an opaque "Internal Server Error" forces the
    caller to guess, and guessing has cost several full evaluation runs. The client
    logs the response body, so the traceback lands in --log-file automatically.
    """
    import traceback
    from fastapi.responses import JSONResponse
    tb = traceback.format_exc()
    print(tb, flush=True)
    return JSONResponse(status_code=500, content={
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": tb.splitlines()[-25:],
    })
