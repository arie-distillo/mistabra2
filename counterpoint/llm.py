"""LLM access. Two backends behind one interface:

  OpenRouterLLM  — real oracle, OpenAI-compatible, temperature 0, JSON-only.
  MockLLM        — deterministic, offline; used by the whole test suite and by
                   `--demo` runs without an API key. It produces *plausible*
                   structured outputs from cheap lexical heuristics so the
                   plumbing (chunking, atomization, caching, matrix arithmetic)
                   can be exercised end-to-end with no network.

The oracle prompts live here so the scorer file stays about arithmetic.
"""
from __future__ import annotations
import json
import re
import time
from typing import Protocol

JSON_RE = re.compile(r"\{.*\}", re.S)


def extract_json(text: str) -> dict:
    if not text or not text.strip():
        raise ValueError("model returned empty or null content")
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = JSON_RE.search(t)
        if m:
            return json.loads(m.group(0))
    raise ValueError(f"no JSON object found in model output: {text[:200]!r}")


class LLM(Protocol):
    def json(self, prompt: str, max_tokens: int = 500) -> dict: ...


# --------------------------------------------------------------------------- real
class OpenRouterLLM:
    def __init__(self, api_key: str, model: str, temperature: float = 0.0,
                 max_retries: int = 4):
        from openai import OpenAI  # imported lazily so tests need no openai pkg
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self._client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key,
                              default_headers={"X-Title": "counterpoint-poc"})

    def json(self, prompt: str, max_tokens: int = 500) -> dict:
        last = None
        for i in range(self.max_retries):
            try:
                kw = dict(model=self.model, temperature=self.temperature,
                          max_tokens=max_tokens,
                          messages=[{"role": "system",
                                     "content": "Reply with a single JSON object. "
                                                "No prose, no markdown fences."},
                                    {"role": "user", "content": prompt}])
                if i == 0:
                    kw["response_format"] = {"type": "json_object"}
                r = self._client.chat.completions.create(**kw)
                content = r.choices[0].message.content
                return extract_json(content)
            except ValueError as e:
                # non-transient: empty/None content or unparseable JSON. Retrying
                # won't help at temperature 0. Fall back to an empty dict so the
                # scorer's own defaults apply, instead of burning retries + sleeps.
                last = e
                break
            except Exception as e:
                # transient (network, rate limit, timeout): back off and retry
                last = e
                time.sleep(1.0 * (i + 1))
        # graceful degradation: let the caller's field defaults take over
        return {}


# --------------------------------------------------------------------------- mock
_STOP = set("the a an of to and or is are was were in on at for with by from this that "
            "it its as be been being their his her they he she we you i".split())


def _tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP and len(w) > 2}


class MockLLM:
    """Deterministic stand-in. Not intelligent — it fabricates *structured,
    plausible* answers so integration tests run without a network. Its lift
    heuristic: lexical overlap between hypothesis and data point drives the
    change label, with a couple of hard-coded polarity cues so a few cells come
    out as genuine support/contradiction rather than all-neutral."""

    NEG_CUES = ("no ", "not ", "without", "within specification", "no signal",
                "no abnormalities", "within normal", "denied", "ruled out",
                "no faults", "stable", "unchanged")
    model = "mock/deterministic-v1"

    def json(self, prompt: str, max_tokens: int = 500) -> dict:
        kind = self._route(prompt)
        if kind == "atomize":
            return self._atomize(prompt)
        if kind == "prior":
            return self._prior(prompt)
        if kind == "relatedness":
            return self._relatedness(prompt)
        if kind == "lift":
            return self._lift(prompt)
        return {}

    def _route(self, p: str) -> str:
        if "Split the passage" in p or "atomic" in p and "propositions" in p:
            return "atomize"
        if "base rate" in p.lower() or "randomly chosen" in p.lower():
            return "prior"
        if "topically related" in p.lower():
            return "relatedness"
        if "change" in p.lower() and "scenario" in p.lower():
            return "lift"
        return "unknown"

    # -- helper to pull a labelled payload string out of a prompt --
    def _field(self, p: str, label: str) -> str:
        m = re.search(rf"{re.escape(label)}\s*(.+)", p)
        return m.group(1).strip() if m else ""

    def _atomize(self, p: str) -> dict:
        m = re.search(r"PASSAGE:\s*(.+)", p, re.S)
        passage = (m.group(1) if m else p).strip()
        parts = re.split(r"(?<=[.!?])\s+", passage)
        pts = [s.strip() for s in parts if len(s.strip()) > 15]
        return {"data_points": pts or [passage[:200]]}

    def _prior(self, p: str) -> dict:
        stmt = self._field(p, "Statement:")
        t = _tokens(stmt)
        base = 0.5
        if any(c in stmt.lower() for c in ("only", "exactly", "precisely", "23-minute",
                                           "14x", "420%", "34 of 96")):
            base = 0.08                       # specific quantitative facts are rare a priori
        elif len(t) > 25:
            base = 0.3
        return {"prior": round(base, 3)}

    def _relatedness(self, p: str) -> dict:
        a = _tokens(self._field(p, "A:"))
        b = _tokens(self._field(p, "B:"))
        j = len(a & b) / max(len(a | b), 1)
        return {"relatedness": round(min(1.0, j * 2.2), 3)}

    def _lift(self, p: str) -> dict:
        h = self._field(p, "learn that this is TRUE --")
        d = self._field(p, "consider this statement:")
        ht, dt = _tokens(h), _tokens(d)
        overlap = len(ht & dt) / max(len(ht | dt), 1)
        neg = any(cue in d.lower() for cue in self.NEG_CUES)
        # crude but deterministic: strong topical overlap + a negation cue -> disfavour;
        # strong overlap without negation -> support; little overlap -> unchanged.
        if overlap < 0.06:
            change = "unchanged"
        elif neg and overlap > 0.12:
            change = "much_less"
        elif neg:
            change = "less"
        elif overlap > 0.28:
            change = "much_more"
        elif overlap > 0.12:
            change = "more"
        else:
            change = "unchanged"
        return {"change": change, "why": f"mock:overlap={overlap:.2f},neg={neg}"}
