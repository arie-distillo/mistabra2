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
from .eval_metrics import compute_metrics, print_metrics_report


def _launch_local_server(port: int, log: StepLogger, embedding_backend: str = "hashing"):
    """Start uvicorn in a subprocess with a mock LLM + in-memory DB, for a
    self-contained local acceptance run that costs nothing and needs no key."""
    env = dict(os.environ)
    env.pop("OPENROUTER_API_KEY", None)              # remove any inherited key
    env["COUNTERPOINT_FORCE_MOCK"] = "1"             # and hard-force the mock, so a
    #                                                  key in .env cannot re-enable
    #                                                  the real LLM via load_dotenv()
    env["COUNTERPOINT_EMBEDDING_BACKEND"] = embedding_backend
    env["COUNTERPOINT_CACHE_DB_PATH"] = ":memory:"
    log.detail(f"launching uvicorn on port {port} "
               f"(mock LLM forced, in-memory DB, embeddings={embedding_backend})")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "counterpoint.app:app",
         "--port", str(port), "--log-level", "warning"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc



def run_eval_over_http(client: ApiClient, log: StepLogger, scenarios: list[int],
                       model_label: str, noise: int = 4):
    """Drive the correctness evaluation entirely over HTTP: for each scenario, reset,
    load it (+noise) via the API, fetch the scored matrix, and compute M1/M2/M3 from
    the response. Returns the per-scenario metrics list (a report, not a verdict)."""
    from .eval_metrics import ScenarioMetrics
    results = []
    for idx in scenarios:
        log.banner(f"EVAL scenario {idx}")
        client.post("/api/reset", params={"in_memory": True})
        r = client.post("/api/load_scenario", json_body={"index": idx, "noise": noise})
        if r.status_code != 200:
            log.detail(f"skipping scenario {idx}: load failed ({r.status_code})")
            continue
        log.detail(f"loaded: {r.json().get('loaded')}")
        log.detail("building scored matrix (this makes the real scoring calls) ...")
        m = client.post("/api/matrix")
        if m.status_code != 200:
            log.detail(f"skipping scenario {idx}: matrix failed ({m.status_code})")
            continue
        metrics = compute_metrics(idx, m.json())
        log._p(f"  → top hypothesis: {metrics.top_hypothesis[:60]}")
        log._p(f"  → M1={metrics.m1} M2={metrics.m2} M3={metrics.m3}"
               + (f"  ! {'; '.join(metrics.notes)}" if metrics.notes else ""))
        results.append(metrics)
    print_metrics_report(results, model_label, log)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="Counterpoint acceptance tests")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--base-url", help="target a running system (local or cloud)")
    g.add_argument("--local", action="store_true",
                   help="launch a throwaway local mock server, test it, tear it down")
    ap.add_argument("--port", type=int, default=8123, help="port for --local")
    ap.add_argument("--json", action="store_true", help="print a JSON summary at the end")
    ap.add_argument("--quiet", action="store_true", help="less verbose step logging")
    ap.add_argument("--log-file", default=None,
                    help="also write the full rich trace to this file (even under --quiet)")
    ap.add_argument("--embedding-backend", default="hashing",
                    choices=["hashing", "tfidf", "sentence_transformers"],
                    help="embedding backend for the --local server "
                         "(default hashing; sentence_transformers exercises the real "
                         "semantic-chunking path but is slower and needs the package)")
    ap.add_argument("--eval", action="store_true",
                    help="after the pass/fail machinery check, also run the correctness "
                         "evaluation (M1/M2/M3) over HTTP. Graded measurements only — "
                         "does NOT affect the exit code.")
    ap.add_argument("--scenario", type=int, default=None,
                    help="with --eval: evaluate only this scenario index (default: all)")
    ap.add_argument("--noise", type=int, default=4,
                    help="with --eval: noise data points to inject per scenario")
    args = ap.parse_args(argv)

    log = StepLogger(verbose=not args.quiet, log_file=args.log_file)
    proc = None
    base_url = args.base_url

    try:
        if args.local:
            proc = _launch_local_server(args.port, log, args.embedding_backend)
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

        # --eval: graded correctness measurements ON TOP of the pass/fail verdict.
        # These print/report but never change the exit code.
        if args.eval:
            from counterpoint.fixtures import all_scenarios
            n = len(all_scenarios())
            scenarios = [args.scenario] if args.scenario is not None else list(range(n))
            log.banner("CORRECTNESS EVALUATION (graded, non-gating)")
            run_eval_over_http(client, log, scenarios,
                               model_label=base_url, noise=args.noise)

        if args.json:
            print(jsonlib.dumps({
                "base_url": base_url,
                "passed": passed,
                "failed": failed,
                "ok": not failed,
            }, indent=2))

        client.close()
        # exit code reflects the MACHINERY verdict only, never the eval metrics
        return 0 if not failed else 1

    finally:
        log.close()
        if proc is not None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    sys.exit(main())
