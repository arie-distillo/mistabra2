"""Pluggable sentence-embedding backends.

  HashingBackend  — deterministic hashed bag-of-words vectors. No deps, no network,
                    fully reproducible. Used by tests and by offline demo runs.
  TfidfBackend    — scikit-learn TF-IDF over the batch. Better semantic grouping
                    than hashing, still offline.
  STBackend       — sentence-transformers (bge-base by default). Production quality;
                    only backend that captures real semantic similarity.

All return L2-normalised row vectors so cosine == dot product.
"""
from __future__ import annotations
import hashlib
import numpy as np


def _l2(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


class HashingBackend:
    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        import re
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        return _l2(out)


class TfidfBackend:
    def encode(self, texts: list[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer
        if len(texts) < 2:
            return _l2(np.ones((len(texts), 1), dtype=np.float32))
        v = TfidfVectorizer(stop_words="english")
        m = v.fit_transform(texts).toarray().astype(np.float32)
        return _l2(m)


class STBackend:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True).astype(np.float32)


def get_backend(name: str, model_name: str = "BAAI/bge-base-en-v1.5"):
    if name in ("sentence_transformers", "st"):   # "st" kept as a back-compat alias
        return STBackend(model_name)
    if name == "tfidf":
        return TfidfBackend()
    return HashingBackend()
