"""Atomize chunks into single-proposition data points.

Why this exists: the lift scorer assumes ONE proposition per data point. A chunk
carrying several propositions gets a meaningless lift (which proposition is being
scored?). Atomization is a prerequisite, not an optimisation. Each data point
keeps a backpointer to its source chunk for citation.
"""
from __future__ import annotations
from .models import Chunk, DataPoint
from .llm import LLM

ATOMIZE_PROMPT = """Split the passage below into atomic, self-contained factual statements.
Rules:
- exactly one proposition per statement (one subject, one claim)
- resolve pronouns and references so each statement stands alone out of context
- preserve specific figures, dates, names, and quantities verbatim
- do not add interpretation, do not infer causes, do not editorialise
- omit pure meta-text (headings, "log entry", document labels) unless it carries a fact

Return only JSON: {"data_points": ["statement 1", "statement 2", ...]}

PASSAGE:
{passage}"""


def atomize_chunk(chunk: Chunk, llm: LLM) -> list[DataPoint]:
    resp = llm.json(ATOMIZE_PROMPT.replace("{passage}", chunk.text), max_tokens=900)
    items = resp.get("data_points", [])
    out = []
    seen = set()
    for t in items:
        t = (t or "").strip()
        if len(t) < 10 or t in seen:
            continue
        seen.add(t)
        out.append(DataPoint.make(chunk.chunk_id, chunk.doc_id, t, meta=dict(chunk.meta)))
    return out
