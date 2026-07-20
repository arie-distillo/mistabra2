"""Orchestration: ingest -> chunk -> atomize -> score grid -> matrix.

Holds a Store, an LLM, a Scorer. Idempotent and cache-backed: re-running over the
same documents/hypotheses re-uses cached atomization implicitly (content-addressed
data points) and cached lift cells.
"""
from __future__ import annotations
from .config import Settings
from .models import Document, Hypothesis
from .store import Store
from .llm import LLM, MockLLM
from .embeddings import get_backend
from .ingest import chunk_document
from .atomize import atomize_chunk
from .scoring import Scorer
from .validation import build_matrix


class Pipeline:
    def __init__(self, settings: Settings | None = None, llm: LLM | None = None,
                 store: Store | None = None):
        self.s = settings or Settings()
        self.store = store or Store(self.s.cache_db_path)
        self.llm = llm or self._default_llm()
        self.backend = get_backend(self.s.embedding_backend, self.s.embedding_model)
        self.scorer = Scorer(self.llm, self.store, self.s)

    def _default_llm(self) -> LLM:
        if self.s.has_api_key:
            from .llm import OpenRouterLLM
            return OpenRouterLLM(self.s.openrouter_api_key, self.s.model,
                                 self.s.llm_temperature, self.s.llm_max_retries)
        return MockLLM()

    # ---- US: add corpus ----
    def add_document(self, name: str, text: str, meta: dict | None = None) -> Document:
        from .models import DataPoint  # local to avoid cycle at import time
        doc_id = f"doc::{abs(hash(name)) % 10**8}"
        doc = Document(doc_id=doc_id, name=name, text=text, meta=meta or {})
        self.store.add_document(doc)
        for chunk in chunk_document(doc, self.s, self.backend):
            self.store.add_chunk(chunk)
            for dp in atomize_chunk(chunk, self.llm):
                self.store.add_data_point(dp)
        return doc

    def add_atomic(self, text: str, meta: dict | None = None):
        """Store `text` as a SINGLE data point, bypassing chunking and atomization.

        For callers whose input is already atomic. Avoids both an LLM call and the
        real damage of re-atomizing: a single proposition gets split into fragments
        that lose their referent and dilute any semantic signal.
        """
        from .models import Document, Chunk, DataPoint
        import hashlib
        meta = meta or {}
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
        doc = Document(doc_id=f"doc::{h}", name=meta.get("source", "atomic"),
                       text=text, meta=dict(meta))
        self.store.add_document(doc)
        ch = Chunk(chunk_id=f"chunk::{h}", doc_id=doc.doc_id, text=text, order=0,
                   meta=dict(meta))
        self.store.add_chunk(ch)
        dp = DataPoint.make(chunk_id=ch.chunk_id, doc_id=doc.doc_id, text=text,
                            meta=dict(meta))
        self.store.add_data_point(dp)
        return {"doc_id": doc.doc_id, "n_data_points": 1, "data_points": [dp]}

    def add_episode(self, text: str, meta: dict | None = None):
        """A pre-atomized single data point (e.g. a YAML 'episode'). Still passes
        through atomization in case the episode bundles several propositions."""
        return self.add_document(name=(meta or {}).get("source", "episode"),
                                 text=text, meta=meta or {})

    # ---- US: add hypotheses ----
    def add_hypothesis(self, text: str, context_summary: str = "",
                       meta: dict | None = None) -> Hypothesis:
        h = Hypothesis.make(text, context_summary, meta)
        self.store.add_hypothesis(h)
        return h

    # ---- US: build the matrix (scores everything, cached) ----
    def matrix(self, progress=None):
        hyps = self.store.hypotheses()
        dps = self.store.data_points()
        cells = self.scorer.grid(hyps, dps, progress=progress,
                                 concurrency=self.s.scoring_concurrency)
        return build_matrix(hyps, dps, cells)
