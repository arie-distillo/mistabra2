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
from .machinery import ALL_CHECKS
from .eval_metrics import compute_metrics, print_metrics_report
from .capability_eval import run_capability_eval


def _launch_local_server(port: int, log: StepLogger, embedding_backend: str = "hashing",
                         model: str | None = None, cache_db: str | None = None):
    """Start uvicorn in a subprocess with a mock LLM + in-memory DB, for a
    self-contained local acceptance run that costs nothing and needs no key."""
    env = dict(os.environ)
    env.pop("OPENROUTER_API_KEY", None)              # remove any inherited key
    env["COUNTERPOINT_FORCE_MOCK"] = "1"             # and hard-force the mock, so a
    #                                                  key in .env cannot re-enable
    #                                                  the real LLM via load_dotenv()
    if model:
        env["ORACLE_MODEL"] = model
        env.pop("COUNTERPOINT_FORCE_MOCK", None)   # a real model was requested
    env["COUNTERPOINT_EMBEDDING_BACKEND"] = embedding_backend
    env["COUNTERPOINT_CACHE_DB_PATH"] = cache_db or ":memory:"
    llm_label = f"model={model}" if model else "mock LLM forced"
    log.detail(f"launching uvicorn on port {port} "
               f"({llm_label}, in-memory DB, embeddings={embedding_backend})")
    # Capture output to a TEMP FILE, not a pipe. A server that dies at import time
    # must be diagnosable — but an unread PIPE deadlocks the child as soon as its
    # 64KB buffer fills, which a model download reliably does. Files never block.
    import tempfile
    log_fh = tempfile.NamedTemporaryFile(mode="w+", suffix=".serverlog",
                                         delete=False, encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "counterpoint.app:app",
         "--port", str(port), "--log-level", "warning"],
        env=env, stdout=log_fh, stderr=subprocess.STDOUT, text=True)
    proc._server_log_path = log_fh.name          # read by _report_server_death
    return proc


def _report_server_death(proc, log):
    """If the launched server exited, surface WHY instead of leaving the caller with
    an unexplained connection failure."""
    if proc is None or proc.poll() is None:
        return False
    out = ""
    try:
        path = getattr(proc, "_server_log_path", None)
        if path:
            with open(path, encoding="utf-8", errors="replace") as fh:
                out = fh.read()
    except Exception:
        pass
    log.always("")
    log.always("!! The local server exited before it could serve requests.")
    log.always("!! Its output was:")
    for line in out.strip().splitlines()[-25:]:
        log.always("     " + line)
    if "sentence_transformers" in out or "sentence-transformers" in out:
        log.always("")
        log.always("!! Hint: the sentence_transformers backend needs the package:")
        log.always("!!     pip install sentence-transformers")
        log.always("!! Or drop --embedding-backend to use the offline 'hashing' backend.")
    return True



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
    ap.add_argument("--model", default=None,
                    help="with --local: ORACLE_MODEL for the launched server, e.g. "
                         "openai/gpt-5. Requires OPENROUTER_API_KEY. Loop the command "
                         "to compare models; the server hosts one model per run.")
    ap.add_argument("--cases-file", metavar="PATH", default=None,
                    help="authored capability cases YAML (e.g. data/phase1_cases.yaml). "
                         "Enables the capability evaluation. Graded, non-gating.")
    ap.add_argument("--capabilities", default="both",
                    choices=["conjunction", "missing", "both"],
                    help="with --cases-file: which capability to exercise "
                         "(default: both)")
    ap.add_argument("--only", metavar="CASE_ID", default=None,
                    help="with --cases-file: run a single case id, for debugging")
    ap.add_argument("--cache-db", default=None, metavar="PATH",
                    help="file-backed score cache for the --local server. Re-running "
                         "the same cases then costs no LLM calls (cells are keyed by "
                         "content hash). Without this, every run re-pays in full.")
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
            proc = _launch_local_server(args.port, log, args.embedding_backend,
                                        args.model, args.cache_db)
            base_url = f"http://127.0.0.1:{args.port}"

        client = ApiClient(base_url=base_url, log=log)
        log.banner(f"ACCEPTANCE RUN against {base_url}")
        # sentence_transformers downloads/loads a model on first use, which can take
        # a minute or more; give it a much longer window than the trivial backends.
        retries = 120 if args.embedding_backend == "sentence_transformers" else 15
        alive = (lambda: proc is None or proc.poll() is None)
        if not wait_for_health(client, retries=retries, delay=1.0, is_alive=alive):
            if _report_server_death(proc, log):
                return 1
            # still alive but never answered: show what it has printed so far
            path = getattr(proc, "_server_log_path", None) if proc else None
            if path:
                log.always("")
                log.always("!! Server did not become ready in time. Its output so far:")
                try:
                    with open(path, encoding="utf-8", errors="replace") as fh:
                        for line in fh.read().strip().splitlines()[-25:]:
                            log.always("     " + line)
                except Exception:
                    pass
            log.check(False, f"deployment reachable at {base_url}")

        passed, failed = [], []
        checks = ALL_CHECKS
        for scenario in checks:
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

        # summary always prints, even in --quiet, and lands in --log-file
        log.always("\n" + "=" * 78)
        log.always("  SUMMARY")
        log.always("=" * 78)
        log.always(f"  target: {base_url}")
        log.always(f"  passed: {len(passed)}/{len(checks)}")
        for f in failed:
            log.always(f"  FAILED  {f['scenario']}: {f['error']}")

        # --eval: graded correctness measurements ON TOP of the pass/fail verdict.
        # These print/report but never change the exit code.
        if args.eval:
            from counterpoint.fixtures import all_scenarios
            n = len(all_scenarios())
            scenarios = [args.scenario] if args.scenario is not None else list(range(n))
            log.banner("CORRECTNESS EVALUATION (graded, non-gating)")
            run_eval_over_http(client, log, scenarios,
                               model_label=(args.model or base_url), noise=args.noise)

        # --capabilities: graded capability evaluation, also non-gating
        if args.cases_file:
            log.banner("CAPABILITY EVALUATION (graded, non-gating)")
            run_capability_eval(client, log, args.cases_file,
                                capability=args.capabilities, only=args.only,
                                keep_cache=bool(args.cache_db))

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
