"""Acceptance-test harness — drives the system through its HTTP API.

Design goals (per the agreed test architecture):
  * Black-box: talks to the system ONLY over HTTP. No imports of product code.
  * Portable: runs against a LOCAL server or a CLOUD deployment — the only
    difference is --base-url.
  * NOT pytest-dependent: this module runs standalone via `python -m tests.acceptance.run`.
    (A thin optional pytest wrapper exists for CI, but is not required.)
  * Verbose by design: every step logs what it did, with what data, and the result,
    so a failure against a remote deployment can be diagnosed from the log alone.

This file provides the harness (client + logging + assertions). The actual test
steps live in scenarios.py; the entry point is run.py.
"""
from __future__ import annotations
import json
import sys
import time
import textwrap
from dataclasses import dataclass, field

import httpx


# --------------------------------------------------------------------------- logging
class StepLogger:
    """Prints a structured, human-readable trace of every request/response and
    every assertion. Also accumulates a machine-readable record for --json output."""

    def __init__(self, verbose: bool = True, log_file: str | None = None):
        self.verbose = verbose
        self.records: list[dict] = []
        self._t0 = time.time()
        # optional persistent sink: the full rich trace is written here regardless
        # of --quiet, so there's a report on disk after the run, not just scrollback.
        self._fh = open(log_file, "w", encoding="utf-8") if log_file else None

    def _p(self, s: str = ""):
        if self.verbose:
            print(s, flush=True)
        if self._fh:
            self._fh.write(s + "\n")
            self._fh.flush()

    def always(self, s: str = ""):
        """Print regardless of --quiet (for summaries and the eval report), and
        still persist to the log file."""
        print(s, flush=True)
        if self._fh:
            self._fh.write(s + "\n")
            self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None

    def banner(self, title: str):
        self._p("\n" + "=" * 78)
        self._p(f"  {title}")
        self._p("=" * 78)

    def step(self, n: str, description: str):
        self._p(f"\n[{n}] {description}")

    def request(self, method: str, url: str, payload=None):
        self._p(f"    → {method} {url}")
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False)
            self._p_split(f"      body: {textwrap.shorten(body, 200)}",
                          f"      body: {body}")

    def _p_split(self, console_text: str, file_text: str):
        """Console gets a readable short form; the log file gets the full text.
        A persisted run is then fully diagnosable without a second mechanism."""
        if self.verbose:
            print(console_text, flush=True)
        if self._fh:
            self._fh.write(file_text + "\n")
            self._fh.flush()

    def response(self, status: int, elapsed_ms: float, body_preview: str):
        self._p(f"    ← {status}  ({elapsed_ms:.0f} ms)")
        if body_preview:
            self._p_split(f"      {textwrap.shorten(body_preview, 240)}",
                          f"      {body_preview}")

    def detail(self, msg: str):
        self._p(f"      · {msg}")

    def check(self, ok: bool, description: str, got=None):
        mark = "PASS" if ok else "FAIL"
        line = f"    [{mark}] {description}"
        if not ok and got is not None:
            line += f"  (got: {textwrap.shorten(str(got), 120)})"
        self._p(line)
        self.records.append({"type": "check", "ok": ok, "desc": description,
                             "got": None if ok else str(got)})
        if not ok:
            raise AssertionError(description + (f" | got: {got}" if got is not None else ""))

    def note(self, msg: str):
        self._p(f"    NOTE: {msg}")
        self.records.append({"type": "note", "msg": msg})


# --------------------------------------------------------------------------- client
@dataclass
class ApiClient:
    """Thin HTTP client with logging. Same interface whether local or cloud."""
    base_url: str
    log: StepLogger
    # Real-model matrix builds take minutes even with server-side concurrency;
    # 120s was silently killing every scoring request.
    timeout: float = 900.0
    _client: httpx.Client = field(init=False)

    def __post_init__(self):
        self._client = httpx.Client(base_url=self.base_url.rstrip("/"),
                                    timeout=self.timeout)

    def call(self, method: str, path: str, json_body=None, params=None) -> httpx.Response:
        self.log.request(method, self.base_url.rstrip("/") + path, json_body or params)
        t = time.time()
        r = self._client.request(method, path, json=json_body, params=params)
        ms = (time.time() - t) * 1000
        preview = ""
        try:
            preview = json.dumps(r.json(), ensure_ascii=False)
        except Exception:
            preview = r.text
        self.log.response(r.status_code, ms, preview)
        return r

    def post(self, path, json_body=None, params=None):
        return self.call("POST", path, json_body, params)

    def get(self, path, params=None):
        return self.call("GET", path, None, params)

    def close(self):
        self._client.close()


# --------------------------------------------------------------------------- health
def wait_for_health(client: ApiClient, retries: int = 10, delay: float = 1.0,
                    is_alive=None) -> bool:
    """Poll until the deployment answers. Uses /docs (always present on FastAPI).

    `is_alive`, if given, is called each round; returning False aborts immediately.
    This stops a long health window from masking a server that died at startup.
    """
    for i in range(retries):
        if is_alive is not None and not is_alive():
            client.log.detail("server process exited during startup")
            return False
        try:
            r = client._client.get("/docs")
            if r.status_code < 500:
                client.log.detail(f"health ok after {i+1} attempt(s)")
                return True
        except Exception as e:
            if i < 3 or i % 10 == 0:      # don't spam for long windows
                client.log.detail(f"health attempt {i+1} failed: {e}")
        time.sleep(delay)
    return False
