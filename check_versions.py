"""Check that the local working copy is internally consistent.

    python check_versions.py

Patching file-by-file across many iterations has repeatedly produced a MIXED state —
e.g. a new json() method with an old __init__, which crashed every LLM call in the
bookkeeping right after a successful request. Symptoms of a mixed state look like
logic bugs, so they get debugged as logic bugs, expensively.

This checks for marker strings that must be present together, and names what is stale.
No network, no LLM, instant.
"""
import sys
from pathlib import Path

# (file, function, markers) — markers must appear INSIDE that function.
# Presence anywhere in the file is not enough: the real failure was a new json()
# body paired with an old __init__, so the marker existed but was never initialised.
SCOPED_CHECKS = [
    ("counterpoint/llm.py", "__init__",
     ["_stats_lock", "_successes", "_failures", "_recent_errors"],
     "attributes json() uses; if __init__ lacks them every LLM call raises "
     "AttributeError after a SUCCESSFUL request"),
]


def _function_body(text: str, name: str) -> str:
    """Crude but sufficient: from 'def name(' to the next line starting with 'def '
    at the same or lower indentation."""
    import re
    m = re.search(rf"\n(\s*)def {re.escape(name)}\s*\(", text)
    if not m:
        return ""
    indent, start = len(m.group(1)), m.end()
    out = []
    for line in text[start:].splitlines():
        if line.strip().startswith("def ") and (len(line) - len(line.lstrip())) <= indent:
            break
        out.append(line)
    return "\n".join(out)


# (file, marker, why it matters)
CHECKS = [
    ("counterpoint/llm.py", "_stats_lock",
     "systemic-vs-scattered failure accounting; without it json() raises AttributeError"),
    ("counterpoint/llm.py", "_recent_errors",
     "captures real error text so failures surface via /api/matrix"),
    ("counterpoint/llm.py", "BUDGET problem",
     "retry-on-empty with a larger budget"),
    ("counterpoint/app.py", "llm_calls_ok",
     "LLM health in the matrix response (silent degradation would be invisible)"),
    ("counterpoint/app.py", "exception_handler",
     "returns the real traceback instead of bare 'Internal Server Error'"),
    ("counterpoint/app.py", "atomize: bool",
     "atomize=false for already-atomic authored cases"),
    ("counterpoint/app.py", "distance_threshold",
     "clustering threshold is sweepable for free"),
    ("counterpoint/app.py", "clear_all",
     "file-backed reset actually empties the DB (else cases accumulate)"),
    ("counterpoint/store.py", "def clear_all",
     "reopening a file DB does not clear it"),
    ("counterpoint/store.py", "joint_cache",
     "conjunction search results are cached"),
    ("counterpoint/pipeline.py", "def add_atomic",
     "stores authored statements verbatim, no re-atomization"),
    ("counterpoint/pipeline.py", "scoring_concurrency",
     "matrix build runs concurrently (serial exceeds any HTTP timeout)"),
    ("counterpoint/residual.py", "distance_threshold",
     "clustering threshold plumbed through"),
    ("counterpoint/residual.py", "small_corpus_threshold",
     "agglomerative fallback below ~20 points (HDBSCAN finds nothing there)"),
    ("counterpoint/scoring.py", "concurrency: int",
     "grid() accepts concurrency"),
    ("counterpoint/atomize.py", "still INFORMATIVE ALONE",
     "anti-over-splitting prompt"),
    ("counterpoint/config.py", "force_mock",
     "mock cannot be re-enabled by a key in .env"),
    ("tests/acceptance/machinery.py", "not over-split",
     "over-splitting fails the machinery check"),
    ("tests/acceptance/capability_eval.py", '"atomize": False',
     "authored cases are not re-atomized"),
    ("tests/acceptance/http_client.py", "_p_split",
     "full response bodies reach --log-file"),
    ("tests/acceptance/run.py", "cases-file",
     "current CLI (--cases-file / --capabilities / --only)"),
]

MISSING_FILES = [
    ("tests/acceptance/journeys.py", "should be DELETED — replaced by machinery.py"),
]


def main():
    root = Path(__file__).parent
    stale, ok = [], 0
    for rel, marker, why in CHECKS:
        p = root / rel
        if not p.exists():
            stale.append((rel, marker, "FILE MISSING", why))
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if marker in text:
            ok += 1
        else:
            stale.append((rel, marker, "stale", why))

    for rel, func, markers, why in SCOPED_CHECKS:
        p = root / rel
        if not p.exists():
            stale.append((rel, func, "FILE MISSING", why))
            continue
        body = _function_body(p.read_text(encoding="utf-8", errors="replace"), func)
        if not body:
            stale.append((rel, f"def {func}", "FUNCTION NOT FOUND", why))
            continue
        missing = [m for m in markers if m not in body]
        if missing:
            stale.append((rel, f"{func}() missing {', '.join(missing)}",
                          "INCONSISTENT", why))
        else:
            ok += 1

    for rel, why in MISSING_FILES:
        if (root / rel).exists():
            stale.append((rel, "-", "SHOULD NOT EXIST", why))

    print(f"checked {len(CHECKS)+len(SCOPED_CHECKS)} markers: {ok} present, {len(stale)} problem(s)\n")
    if not stale:
        print("Working copy looks consistent.")
        return 0
    by_file = {}
    for rel, marker, status, why in stale:
        by_file.setdefault(rel, []).append((marker, status, why))
    print("PROBLEMS — these files are out of date or inconsistent:\n")
    for rel, items in by_file.items():
        print(f"  {rel}")
        for marker, status, why in items:
            print(f"      [{status}] missing marker: {marker!r}")
            print(f"               {why}")
        print()
    print("Any mixed state can produce failures that look like logic bugs.")
    print("Update the files listed above before interpreting any run.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
