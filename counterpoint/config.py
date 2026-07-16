"""Central configuration and the lift scale.

The lift scale is the single most load-bearing design decision in the scorer,
settled empirically in the Tier-1 oracle experiments:

  * We elicit the RATIO (a multiplier) directly, never an absolute probability
    that we then divide by a prior. Eliciting absolute-prob / prior inverted the
    lift whenever the prior was extreme (a documented v1 failure).
  * "unchanged" maps to exactly 1.0 — the anchor that separates the topical axis
    from the logical axis. A data point can be highly related to a hypothesis and
    still have lift 1.0 (no logical bearing).
"""
from __future__ import annotations
import os
from dataclasses import dataclass

# Load a .env file (if present) into the environment before we read any vars.
# Real secrets like OPENROUTER_API_KEY live in .env, which is gitignored.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # dotenv is optional; env vars still work without it
    pass

# Ordinal change label -> lift multiplier. "guaranteed" resolves at runtime to
# min(0.97 / prior, cap) because P(D|H) cannot exceed 1.
LIFT_SCALE = {
    "ruled_out": 0.02,
    "much_less": 0.20,
    "less": 0.50,
    "unchanged": 1.00,
    "more": 2.00,
    "much_more": 5.00,
    "guaranteed": None,
}
GUARANTEED_PCH = 0.97          # P(D|H) used for the "guaranteed" bucket
LIFT_CAP = 20.0                # clamp extreme lifts (keeps log-lift bounded)

# Verdict bands over log-lift. Symmetric, with a neutral dead-zone around 0 that
# encodes "topically present but logically inert".
VERDICT_BANDS = [
    (-float("inf"), -1.0, "strongly_contradicts"),
    (-1.0, -0.30, "contradicts"),
    (-0.30, 0.30, "neutral"),
    (0.30, 1.0, "supports"),
    (1.0, float("inf"), "strongly_supports"),
]


@dataclass
class Settings:
    openrouter_api_key: str = os.environ.get("OPENROUTER_API_KEY", "")
    model: str = os.environ.get("ORACLE_MODEL", "anthropic/claude-sonnet-4.5")
    # Path to the SQLite file holding the corpus and the lift-score cache.
    cache_db_path: str = os.environ.get("COUNTERPOINT_CACHE_DB_PATH", "counterpoint.db")
    # Which sentence-embedding strategy to use for chunking / clustering:
    #   hashing               — offline, deterministic bag-of-words (tests, no deps)
    #   tfidf                 — offline scikit-learn TF-IDF (lexical, better grouping)
    #   sentence_transformers — real semantic embeddings (production quality)
    embedding_backend: str = os.environ.get("COUNTERPOINT_EMBEDDING_BACKEND", "hashing")
    # Model name used only when embedding_backend == "sentence_transformers".
    embedding_model: str = os.environ.get("COUNTERPOINT_EMBEDDING_MODEL",
                                          "BAAI/bge-base-en-v1.5")
    # semantic chunking
    chunk_similarity_percentile: int = 25   # split at similarity troughs below this
    chunk_min_sentences: int = 1
    chunk_max_sentences: int = 8
    # scoring
    llm_temperature: float = 0.0
    llm_max_retries: int = 4

    @property
    def has_api_key(self) -> bool:
        return bool(self.openrouter_api_key)


def verdict_for(log_lift: float) -> str:
    for lo, hi, name in VERDICT_BANDS:
        if lo <= log_lift < hi:
            return name
    return "neutral"
