"""SQLite persistence. The file holds two categorically different kinds of data:

  PERSISTENT CORPUS STATE (your ingested data, not a cache):
    documents, chunks, data_points, hypotheses

  SCORE CACHE (expensive LLM outputs, safe to delete and regenerate):
    lift_cache   keyed (hypothesis_id, data_point_id, model) -> lift, verdict, rationale
    prior_cache  keyed (data_point_id, model)               -> P(D)

The score cache is why re-building the matrix costs zero LLM calls. Because data
points and hypotheses are content-addressed (id = hash of text), re-ingesting the
same sentence transparently reuses its cached scores across runs and documents.
"""
from __future__ import annotations
import json
import sqlite3
from typing import Optional
from .models import Document, Chunk, DataPoint, Hypothesis, LiftCell


def _dumps(obj) -> str:
    # meta from YAML can contain datetime.date etc. — serialise defensively
    return json.dumps(obj, default=str)

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents(
  doc_id TEXT PRIMARY KEY, name TEXT, text TEXT, meta TEXT);
CREATE TABLE IF NOT EXISTS chunks(
  chunk_id TEXT PRIMARY KEY, doc_id TEXT, text TEXT, ord INTEGER, meta TEXT);
CREATE TABLE IF NOT EXISTS data_points(
  dp_id TEXT PRIMARY KEY, chunk_id TEXT, doc_id TEXT, text TEXT, meta TEXT, prior REAL);
CREATE TABLE IF NOT EXISTS hypotheses(
  hyp_id TEXT PRIMARY KEY, text TEXT, context_summary TEXT, meta TEXT);
CREATE TABLE IF NOT EXISTS lift_cache(
  hyp_id TEXT, dp_id TEXT, model TEXT, change TEXT, lift REAL, log_lift REAL,
  relatedness REAL, verdict TEXT, rationale TEXT,
  PRIMARY KEY (hyp_id, dp_id, model));
CREATE TABLE IF NOT EXISTS prior_cache(
  dp_id TEXT, model TEXT, prior REAL, PRIMARY KEY (dp_id, model));
"""


class Store:
    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def close(self):
        self.db.close()

    # ---- corpus ----
    def add_document(self, d: Document):
        self.db.execute("INSERT OR REPLACE INTO documents VALUES(?,?,?,?)",
                        (d.doc_id, d.name, d.text, _dumps(d.meta)))
        self.db.commit()

    def add_chunk(self, c: Chunk):
        self.db.execute("INSERT OR REPLACE INTO chunks VALUES(?,?,?,?,?)",
                        (c.chunk_id, c.doc_id, c.text, c.order, _dumps(c.meta)))
        self.db.commit()

    def add_data_point(self, dp: DataPoint):
        self.db.execute("INSERT OR REPLACE INTO data_points VALUES(?,?,?,?,?,?)",
                        (dp.dp_id, dp.chunk_id, dp.doc_id, dp.text,
                         _dumps(dp.meta), dp.prior))
        self.db.commit()

    def add_hypothesis(self, h: Hypothesis):
        self.db.execute("INSERT OR REPLACE INTO hypotheses VALUES(?,?,?,?)",
                        (h.hyp_id, h.text, h.context_summary, _dumps(h.meta)))
        self.db.commit()

    def data_points(self) -> list[DataPoint]:
        rows = self.db.execute("SELECT * FROM data_points ORDER BY doc_id, dp_id")
        return [DataPoint(r["dp_id"], r["chunk_id"], r["doc_id"], r["text"],
                          json.loads(r["meta"]), r["prior"]) for r in rows]

    def hypotheses(self) -> list[Hypothesis]:
        rows = self.db.execute("SELECT * FROM hypotheses ORDER BY hyp_id")
        return [Hypothesis(r["hyp_id"], r["text"], r["context_summary"],
                           json.loads(r["meta"])) for r in rows]

    def documents(self) -> list[Document]:
        rows = self.db.execute("SELECT * FROM documents ORDER BY doc_id")
        return [Document(r["doc_id"], r["name"], r["text"], json.loads(r["meta"]))
                for r in rows]

    def chunk(self, chunk_id: str) -> Optional[Chunk]:
        r = self.db.execute("SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
        return Chunk(r["chunk_id"], r["doc_id"], r["text"], r["ord"],
                     json.loads(r["meta"])) if r else None

    # ---- caches ----
    def get_prior(self, dp_id: str, model: str) -> Optional[float]:
        r = self.db.execute("SELECT prior FROM prior_cache WHERE dp_id=? AND model=?",
                            (dp_id, model)).fetchone()
        return r["prior"] if r else None

    def put_prior(self, dp_id: str, model: str, prior: float):
        self.db.execute("INSERT OR REPLACE INTO prior_cache VALUES(?,?,?)",
                        (dp_id, model, prior))
        self.db.commit()

    def get_cell(self, hyp_id: str, dp_id: str, model: str) -> Optional[LiftCell]:
        r = self.db.execute(
            "SELECT * FROM lift_cache WHERE hyp_id=? AND dp_id=? AND model=?",
            (hyp_id, dp_id, model)).fetchone()
        if not r:
            return None
        return LiftCell(r["hyp_id"], r["dp_id"], r["change"], r["lift"], r["log_lift"],
                        r["relatedness"], r["verdict"], r["rationale"], r["model"])

    def put_cell(self, c: LiftCell):
        self.db.execute("INSERT OR REPLACE INTO lift_cache VALUES(?,?,?,?,?,?,?,?,?)",
                        (c.hyp_id, c.dp_id, c.model, c.change, c.lift, c.log_lift,
                         c.relatedness, c.verdict, c.rationale))
        self.db.commit()
