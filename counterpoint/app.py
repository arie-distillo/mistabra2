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
def reset(in_memory: bool = True):
    """Clear all corpus and hypotheses (and, for :memory:, the score cache).
    A dev/test affordance so the system can be driven from an empty state purely
    through the API. `in_memory=true` swaps to a throwaway in-memory DB."""
    global pipeline
    pipeline = _new_pipeline(in_memory=in_memory)
    return {"status": "reset", "in_memory": in_memory}


# --------------------------------------------------------------- API routes
@app.post("/api/documents")
def add_document(d: DocIn):
    doc = pipeline.add_document(d.name, d.text, d.meta)
    dps = [dp for dp in pipeline.store.data_points() if dp.doc_id == doc.doc_id]
    return {"doc_id": doc.doc_id, "n_data_points": len(dps),
            "data_points": [dp.text for dp in dps]}


@app.post("/api/episodes")
def add_episode(e: EpisodeIn):
    doc = pipeline.add_episode(e.content, e.meta)
    dps = [dp for dp in pipeline.store.data_points() if dp.doc_id == doc.doc_id]
    return {"doc_id": doc.doc_id, "n_data_points": len(dps)}


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
                         "diagnosticity": round(m.diagnosticity.get(dp.dp_id, 0.0), 3)}
                        for dp in m.data_points],
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
