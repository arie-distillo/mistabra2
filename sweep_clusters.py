"""Sweep the residual-clustering threshold for one case. FREE — makes no LLM calls
for the clustering itself, and reuses cached scores for the matrix.

    python sweep_clusters.py --case miss_01
    python sweep_clusters.py --case miss_01 --cache-db counterpoint_cache.db
    python sweep_clusters.py --case miss_01 --thresholds 0.6,0.5,0.45,0.4,0.35,0.3

No server, no curl. Runs the pipeline in-process, then re-clusters the SAME residual
set at each threshold and prints cluster membership in the case's own d1/d2/... ids,
marking planted members with *.

The question this answers: can the embedding model see that the planted cluster
belongs together? If no threshold separates it from the distractors, the limitation is
the encoder, not the clustering algorithm — a different fix entirely.
"""
import argparse
import sys
from pathlib import Path

import yaml

from counterpoint.config import Settings
from counterpoint.store import Store
from counterpoint.pipeline import Pipeline
from counterpoint.residual import find_residuals, cluster_residuals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="miss_01")
    ap.add_argument("--cases-file", default="data/phase1_cases.yaml")
    ap.add_argument("--cache-db", default=None,
                    help="reuse cached scores so the matrix costs nothing")
    ap.add_argument("--embedding-backend", default="sentence_transformers")
    ap.add_argument("--model", default=None)
    ap.add_argument("--thresholds", default=None,
                    help="comma-separated; default is auto-chosen from the measured "
                         "distance range, which is where any signal must lie")
    ap.add_argument("--min-cluster-sizes", default="3,2",
                    help="also try smaller clusters: a 4-member planted group can "
                         "fail to form at min_cluster_size=3 if one member is loose")
    ap.add_argument("--epsilon", type=float, default=0.15)
    ap.add_argument("--min-cluster-size", type=int, default=3)
    args = ap.parse_args()

    cases = yaml.safe_load(Path(args.cases_file).read_text(encoding="utf-8"))
    case = next((c for c in cases if c["id"] == args.case), None)
    if case is None:
        print(f"case '{args.case}' not found in {args.cases_file}")
        return 1
    planted = set(case["ground_truth"].get("cluster_members") or [])

    s = Settings(embedding_backend=args.embedding_backend)
    if args.cache_db:
        s.cache_db_path = args.cache_db
    if args.model:
        s.model = args.model
    pipe = Pipeline(s, store=Store(args.cache_db or ":memory:"))
    pipe.store.clear_corpus()

    print(f"case      : {case['id']}  ({case.get('domain','?')})")
    print(f"embeddings: {args.embedding_backend}")
    print(f"planted   : {sorted(planted)}")
    print()

    # load verbatim — these are authored atomic statements, never re-atomize
    for dp in case["data_points"]:
        pipe.add_atomic(dp["text"], meta={"source": dp["id"], "credibility": 1.0})
    for h in (case.get("given_hypotheses") or [case.get("hypothesis")]):
        if h:
            pipe.add_hypothesis(h)

    print("scoring matrix (cached cells are free) ...", flush=True)
    m = pipe.matrix()
    src = {dp.dp_id: dp.meta.get("source", dp.dp_id) for dp in m.data_points}

    residuals = find_residuals(m.hypotheses, m.data_points, m.cells, args.epsilon)
    print(f"residuals : {len(residuals)}/{len(m.data_points)} "
          f"-> {sorted(src[d.dp_id] for d in residuals)}")
    missing = planted - {src[d.dp_id] for d in residuals}
    if missing:
        print(f"  NOTE: planted members NOT in the residual set: {sorted(missing)}")
        print("        (they were explained by a given hypothesis, or their scoring")
        print("         call failed and defaulted)")
    print()

    # Direct evidence, independent of any clustering parameter: is the planted group
    # actually closer together than it is to the distractors? If not, no threshold can
    # separate them and the limitation is the encoder.
    import numpy as np
    r_ids = [src[d.dp_id] for d in residuals]
    emb = pipe.backend.encode([d.text for d in residuals])
    D = 1.0 - (emb @ emb.T)
    P = [i for i, x in enumerate(r_ids) if x in planted]
    Q = [i for i, x in enumerate(r_ids) if x not in planted]
    def mean_of(pairs):
        v = [D[i][j] for i, j in pairs]
        return float(np.mean(v)) if v else float("nan")
    pp = mean_of([(i, j) for i in P for j in P if i < j])
    pq = mean_of([(i, j) for i in P for j in Q])
    qq = mean_of([(i, j) for i in Q for j in Q if i < j])
    print("PAIRWISE COSINE DISTANCE (lower = more similar)")
    print(f"  planted <-> planted     : {pp:.4f}   <- must be clearly LOWEST")
    print(f"  planted <-> distractor  : {pq:.4f}")
    print(f"  distractor<->distractor : {qq:.4f}")
    sep = pq - pp
    pp_v = [D[i][j] for i in P for j in P if i < j]
    pq_v = [D[i][j] for i in P for j in Q]
    # Cohen's d: a raw gap is meaningless without the spread it sits in.
    import math
    sd = math.sqrt((np.var(pp_v) + np.var(pq_v)) / 2) if pp_v and pq_v else 0.0
    d = (pq - pp) / sd if sd else 0.0
    print(f"  separation (pq - pp)    : {sep:+.4f}")
    print(f"  effect size (Cohen's d) : {d:+.2f}   "
          f"(0.2 small / 0.5 medium / 0.8 large)")
    if d < 0.2:
        print("  VERDICT: no usable separation. This is the ENCODER, not a parameter.")
    elif d < 0.8:
        print("  VERDICT: WEAK but real separation. A threshold may partly recover the")
        print("           cluster; expect imperfect purity. Worth sweeping finely.")
    else:
        print("  VERDICT: strong separation; a threshold between pp and pq should work.")
    print()

    if args.thresholds:
        ts = [float(x) for x in args.thresholds.split(",")]
    else:
        # Sweep across the range the distances actually occupy. The previous fixed
        # ladder stepped 0.60 -> 0.50 and skipped the entire signal region.
        lo, hi = min(pp, qq, pq) - 0.03, max(pp, qq, pq) + 0.03
        ts = [round(lo + (hi - lo) * k / 14, 4) for k in range(15)]
    mcs_list = [int(x) for x in args.min_cluster_sizes.split(",")]

    print(f"{'thresh':>7s} {'mcs':>4s} {'clusters':>8s}  composition (* = planted)")
    print("-" * 78)
    best = None
    for mcs in mcs_list:
      for t in ts:
        clusters = cluster_residuals(residuals, pipe.backend,
                                     min_cluster_size=mcs,
                                     small_corpus_threshold=10_000,  # force agglomerative
                                     distance_threshold=t)
        parts = []
        score = None
        for c in clusters:
            ids = [src[i] for i in c.dp_ids]
            marked = " ".join((f"{i}*" if i in planted else i) for i in sorted(ids))
            got = set(ids) & planted
            rec = len(got) / max(len(planted), 1)
            pur = len(got) / max(len(ids), 1)
            parts.append(f"[{marked}] r={rec:.2f} p={pur:.2f}")
            f1 = 0 if (rec + pur) == 0 else 2 * rec * pur / (rec + pur)
            score = max(score or 0, f1)
        print(f"{t:>7.3f} {mcs:>4d} {len(clusters):>8d}  "
              + ("  ".join(parts) if parts else "(none)"))
        if score and (best is None or score > best[1]):
            best = ((t, mcs), score)
    print("-" * 74)
    if best:
        print(f"best separation at distance_threshold={best[0][0]} "
              f"min_cluster_size={best[0][1]} (F1={best[1]:.2f})")
        if best[1] < 0.7:
            print("F1 below 0.7 at every threshold: the planted cluster is not cleanly")
            print("separable with this encoder. That is an EMBEDDING limitation, not a")
            print("clustering-parameter one.")
    else:
        print("no clusters formed at any threshold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
