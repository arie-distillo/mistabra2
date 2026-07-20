"""Atomize chunks into single-proposition data points.

Why this exists: the lift scorer assumes ONE proposition per data point. A chunk
carrying several propositions gets a meaningless lift (which proposition is being
scored?). Atomization is a prerequisite, not an optimisation. Each data point
keeps a backpointer to its source chunk for citation.
"""
from __future__ import annotations
from .models import Chunk, DataPoint
from .llm import LLM

ATOMIZE_PROMPT = """Split the passage into self-contained factual statements.

Each statement must be the smallest unit that is still INFORMATIVE ALONE. Prefer fewer,
complete statements. Splitting too finely destroys facts.

Keep together in ONE statement:
- a measurement and what it is compared against
  ("pressure dropped below the 15 Pa threshold" = ONE statement)
- a fact and its qualifiers (quantity, unit, time, location, entity)
- a finding and what was found
  ("inspection revealed the valve was blocked" -> "The valve was blocked")

Split only genuinely separate facts: different events, entities, or times.

Also:
- resolve pronouns so each statement stands alone
- keep figures, dates, names, quantities verbatim
- no interpretation, no inferred causes
- drop meta-text and anything uninformative by itself

Return only JSON: {"data_points": ["statement 1", "statement 2", ...]}

PASSAGE:
{passage}"""


def atomize_chunk(chunk: Chunk, llm: LLM) -> list[DataPoint]:
    resp = llm.json(ATOMIZE_PROMPT.replace("{passage}", chunk.text), max_tokens=2500)
    # 2500, not 900: the prompt asks the model to re-read and merge over-split
    # statements, which invites extended reasoning. A reasoning-capable model can
    # spend a small budget entirely on thinking and return EMPTY content — which
    # surfaces as an atomization failure, not as a smaller output.
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
