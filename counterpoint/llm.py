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
    # After this many CONSECUTIVE total failures, stop degrading quietly and raise.
    # An isolated bad response should not abort a long run; a systemic fault should.
    FAIL_FAST_AFTER = 3

    def __init__(self, api_key: str, model: str, temperature: float = 0.0,
                 max_retries: int = 4, timeout: float = 60.0):
        import threading
        from openai import OpenAI  # imported lazily so tests need no openai pkg
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        # Failure accounting must distinguish SYSTEMIC faults (bad key, wrong model
        # id: nothing ever succeeds) from SCATTERED ones (an occasional empty
        # response). "N consecutive failures" is a poor proxy under concurrency:
        # with 8 worker threads, unrelated transient failures interleave and look
        # consecutive, aborting a run that is mostly succeeding.
        self._stats_lock = threading.Lock()
        self._successes = 0
        self._failures = 0
        # Keep the actual error text. Printing to stderr is not enough: the server's
        # stderr does not reach the client log, so failures had to be inferred from
        # timing. These surface via /api/matrix.
        self._recent_errors: list[str] = []
        # EXPLICIT timeout: the OpenAI SDK defaults to 600s, so a blocked connection
        # (firewall, proxy, DNS blackhole) hangs for ten minutes per attempt with no
        # output at all. 60s is generous for a single completion.
        self._client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key,
                              timeout=timeout, max_retries=0,
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
                out = extract_json(content)
                with self._stats_lock:
                    self._successes += 1
                return out
            except ValueError as e:
                # Empty content is NOT reliably a prompt or budget problem: the same
                # prompts succeed 6/6 when issued sequentially and fail ~17% of the
                # time at concurrency 8. That pattern says load, not content — the
                # provider returns an empty body instead of a 429. So back off and
                # retry like any transient fault, while also widening the budget in
                # case this particular call really was truncated.
                last = e
                if i < self.max_retries - 1:
                    max_tokens = max(max_tokens * 2, 1200)
                    time.sleep(0.5 * (i + 1))
                    continue
                break
            except Exception as e:
                # transient (network, rate limit, timeout): back off and retry
                last = e
                time.sleep(1.0 * (i + 1))

        # Every attempt failed for this call.
        import sys
        with self._stats_lock:
            self._failures += 1
            failures, successes = self._failures, self._successes
            msg = f"{type(last).__name__}: {last}"
            if len(self._recent_errors) < 8:
                self._recent_errors.append(msg)
        print(f"[LLM ERROR] {type(last).__name__}: {last} "
              f"(ok={successes} failed={failures})", file=sys.stderr, flush=True)
        # Systemic ONLY if nothing has ever succeeded. Once any call has worked, the
        # credentials/model/network are fine and a stray failure should degrade
        # gracefully to caller defaults rather than abort the whole run.
        if successes == 0 and failures >= self.FAIL_FAST_AFTER:
            raise RuntimeError(
                f"{failures} LLM calls failed with no successes against model "
                f"'{self.model}'. Last error: {type(last).__name__}: {last}. "
                f"Check OPENROUTER_API_KEY, the model id, and network access."
            ) from last
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
        if kind == "abduce":
            return self._abduce(prompt)
        if kind == "joint_lift":
            return self._joint_lift(prompt)
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
        if "Propose ONE single hypothesis" in p:
            return "abduce"
        if "taken TOGETHER as a single combined fact" in p:
            return "joint_lift"
        if "base rate" in p.lower() or "randomly chosen" in p.lower():
            return "prior"
        if "topically related" in p.lower():
            return "relatedness"
        if "change" in p.lower() and "scenario" in p.lower():
            return "lift"
        return "unknown"

    def _abduce(self, p: str) -> dict:
        """Deterministic stand-in: name the most frequent salient tokens in the
        cluster. Not intelligent — it exists so the residual pipeline can be
        exercised end-to-end offline."""
        import collections
        lines = [l.strip("  - ") for l in p.splitlines() if l.strip().startswith("- ")]
        toks = collections.Counter()
        for l in lines:
            toks.update(_tokens(l))
        top = [w for w, _ in toks.most_common(4)]
        return {"hypothesis": ("A common underlying factor involving "
                               + ", ".join(top) + " explains these observations."),
                "why": "mock: most frequent shared terms"}

    def _joint_lift(self, p: str) -> dict:
        """Deterministic stand-in for conjunction scoring. Uses overlap between the
        hypothesis and the combined block, with a crude tension cue: if the block
        contains both a negation-ish statement and a supporting one, report a drop."""
        h = self._field(p, "learn that this is TRUE --")
        lines = [l.strip("  - ") for l in p.splitlines() if l.strip().startswith("- ")]
        ht = _tokens(h)
        neg = sum(any(c in l.lower() for c in self.NEG_CUES) for l in lines)
        pos = sum(len(ht & _tokens(l)) > 0 for l in lines)
        if neg and pos:
            change = "much_less"          # tension between members
        elif neg:
            change = "less"
        elif pos >= 2:
            change = "more"
        else:
            change = "unchanged"
        return {"change": change, "why": f"mock joint: neg={neg} pos={pos}"}

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
