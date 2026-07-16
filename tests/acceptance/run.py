"""Acceptance-test runner — standalone entry point (no pytest required).

    # against a cloud deployment
    python -m tests.acceptance.run --base-url https://your-app.up.railway.app

    # against a local server you already started
    python -m tests.acceptance.run --base-url http://127.0.0.1:8000

    # launch a throwaway local server automatically, test it, tear it down
    python -m tests.acceptance.run --local

    # machine-readable summary for CI
    python -m tests.acceptance.run --local --json

Exit code 0 = all scenarios passed, 1 = a failure (with a full log above it).
"""
from __future__ import annotations
import argparse
import json as jsonlib
import os
import signal
import subprocess
import sys
import time

from .http_client import ApiClient, StepLogger, wait_for_health
from .journeys import ALL_JOURNEYS


def _launch_local_server(port: int, log: StepLogger):
    """Start uvicorn in a subprocess with a mock LLM + in-memory DB, for a
    self-contained local acceptance run that costs nothing and needs no key."""
    env = dict(os.environ)
    env.pop("OPENROUTER_API_KEY", None)              # force MockLLM
    env["COUNTERPOINT_EMBEDDING_BACKEND"] = "hashing"
    env["COUNTERPOINT_CACHE_DB_PATH"] = ":memory:"
    log.detail(f"launching uvicorn on port {port} (mock LLM, in-memory DB)")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "counterpoint.app:app",
         "--port", str(port), "--log-level", "warning"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc


def main(argv=None):
    ap = argparse.ArgumentParser(description="Counterpoint acceptance tests")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--base-url", help="target a running system (local or cloud)")
    g.add_argument("--local", action="store_true",
                   help="launch a throwaway local mock server, test it, tear it down")
    ap.add_argument("--port", type=int, default=8123, help="port for --local")
    ap.add_argument("--json", action="store_true", help="print a JSON summary at the end")
    ap.add_argument("--quiet", action="store_true", help="less verbose step logging")
    args = ap.parse_args(argv)

    log = StepLogger(verbose=not args.quiet)
    proc = None
    base_url = args.base_url

    try:
        if args.local:
            proc = _launch_local_server(args.port, log)
            base_url = f"http://127.0.0.1:{args.port}"

        client = ApiClient(base_url=base_url, log=log)
        log.banner(f"ACCEPTANCE RUN against {base_url}")
        if not wait_for_health(client, retries=15, delay=1.0):
            log.check(False, f"deployment reachable at {base_url}")

        passed, failed = [], []
        for scenario in ALL_JOURNEYS:
            name = scenario.__name__
            try:
                scenario(client, log)
                passed.append(name)
            except AssertionError as e:
                failed.append({"scenario": name, "error": str(e)})
                log._p(f"\n    !!! SCENARIO FAILED: {name}: {e}")
            except Exception as e:  # unexpected (transport, 500, etc.)
                failed.append({"scenario": name, "error": f"unexpected: {e}"})
                log._p(f"\n    !!! SCENARIO ERROR: {name}: {e}")

        # summary always prints, even in --quiet
        print("\n" + "=" * 78)
        print("  SUMMARY")
        print("=" * 78)
        print(f"  target: {base_url}")
        print(f"  passed: {len(passed)}/{len(ALL_JOURNEYS)}")
        for f in failed:
            print(f"  FAILED  {f['scenario']}: {f['error']}")

        if args.json:
            print(jsonlib.dumps({
                "base_url": base_url,
                "passed": passed,
                "failed": failed,
                "ok": not failed,
            }, indent=2))

        client.close()
        return 0 if not failed else 1

    finally:
        if proc is not None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    sys.exit(main())
