"""Ingest documents and split them into semantically-coherent chunks.

Semantic chunking (not fixed-size): we embed each sentence, walk the sequence,
and cut where the similarity between consecutive sentences drops into the lower
tail of the distribution (a trough). This keeps a coherent passage together and
starts a new chunk when the topic shifts — which matters here because the next
stage (atomization) works best on topically-uniform passages, and because the
data-point→chunk backpointer is what the UI cites.

Reference: the percentile-of-adjacent-similarity method (Kamradt-style).
"""
from __future__ import annotations
import re
import numpy as np
from .models import Document, Chunk, DataPoint  # noqa: F401
from .config import Settings
from .embeddings import get_backend

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    sents = _SENT_SPLIT.split(text)
    return [s.strip() for s in sents if s.strip()]


def semantic_chunks(sentences: list[str], backend, settings: Settings) -> list[str]:
    """Group sentences into chunks at similarity troughs."""
    n = len(sentences)
    if n <= settings.chunk_min_sentences:
        return [" ".join(sentences)] if sentences else []
    emb = backend.encode(sentences)
    # cosine similarity between consecutive sentences (vectors are normalised)
    adj = np.array([float(emb[i] @ emb[i + 1]) for i in range(n - 1)])
    if len(adj) == 0:
        return [" ".join(sentences)]
    # a "boundary" is a trough: adjacent similarity below the low percentile
    thresh = np.percentile(adj, settings.chunk_similarity_percentile)
    chunks, cur = [], [sentences[0]]
    for i in range(1, n):
        boundary = adj[i - 1] <= thresh
        too_long = len(cur) >= settings.chunk_max_sentences
        if (boundary or too_long) and len(cur) >= settings.chunk_min_sentences:
            chunks.append(" ".join(cur))
            cur = [sentences[i]]
        else:
            cur.append(sentences[i])
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def chunk_document(doc: Document, settings: Settings, backend=None) -> list[Chunk]:
    backend = backend or get_backend(settings.embedding_backend, settings.embedding_model)
    sents = split_sentences(doc.text)
    texts = semantic_chunks(sents, backend, settings)
    out = []
    for i, t in enumerate(texts):
        cid = f"{doc.doc_id}::c{i}"
        out.append(Chunk(chunk_id=cid, doc_id=doc.doc_id, text=t, order=i, meta=dict(doc.meta)))
    return out
