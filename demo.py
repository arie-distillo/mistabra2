"""Offline demo: load a scenario (+noise), build the matrix, print it.

    python demo.py                 # burnt-bread, mock LLM, no noise
    python demo.py --scenario 3 --noise 4 --duplicates 1
    OPENROUTER_API_KEY=... python demo.py   # uses the real oracle

Prints the ranked hypotheses, the most-diagnostic data points, and — for the
top and bottom hypothesis — the single most contradicting / supporting data
point with its citation.
"""
import argparse
from counterpoint.config import Settings
from counterpoint.pipeline import Pipeline
from counterpoint.store import Store
from counterpoint.fixtures import all_scenarios, inject_noise, load_into_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=int, default=0)
    ap.add_argument("--noise", type=int, default=0)
    ap.add_argument("--duplicates", type=int, default=0)
    ap.add_argument("--use-llm", action="store_true",
                    help="use the real OpenRouter LLM (default: fast deterministic mock)")
    args = ap.parse_args()

    # Default to the mock so the demo is fast and free even if a key sits in .env.
    settings = Settings(cache_db_path=":memory:")
    if not args.use_llm:
        settings.force_mock = True
    pipe = Pipeline(settings, store=Store(":memory:"))
    print(f"LLM: {getattr(pipe.llm, 'model', '?')}   embed: {settings.embedding_backend}\n")

    scs = all_scenarios()
    sc = scs[args.scenario]
    if args.noise or args.duplicates:
        sc = inject_noise(sc, n=args.noise, duplicate=args.duplicates)
    print(f"scenario: {sc.name}\n{len(sc.episodes)} episodes, "
          f"{len(sc.hypotheses)} hypotheses\n")
    load_into_pipeline(pipe, sc)

    m = pipe.matrix()
    print("=== hypotheses, ranked by score ===")
    for i, r in enumerate(m.results, 1):
        print(f"  {i}. [{r.score:+6.2f}] {r.verdict:20s} "
              f"(+{r.n_supporting}/-{r.n_contradicting}/~{r.n_neutral})  "
              f"{r.hypothesis.text[:70]}")

    print("\n=== most diagnostic data points ===")
    dp_by = {dp.dp_id: dp for dp in m.data_points}
    for dp_id, d in sorted(m.diagnosticity.items(), key=lambda kv: -kv[1])[:5]:
        print(f"  diag={d:5.2f}  {dp_by[dp_id].text[:80]}")

    print("\n=== least diagnostic (candidate noise / inert) ===")
    for dp_id, d in sorted(m.diagnosticity.items(), key=lambda kv: kv[1])[:5]:
        noise = dp_by[dp_id].meta.get("noise", "")
        tag = f"  [{noise}]" if noise else ""
        print(f"  diag={d:5.2f}  {dp_by[dp_id].text[:70]}{tag}")

    top, bottom = m.results[0], m.results[-1]
    if top.best_cell:
        print(f"\ntop hypothesis best evidence:  "
              f"{dp_by[top.best_cell.dp_id].text[:80]}")
    if bottom.worst_cell:
        print(f"bottom hypothesis worst evidence: "
              f"{dp_by[bottom.worst_cell.dp_id].text[:80]}")


if __name__ == "__main__":
    main()
