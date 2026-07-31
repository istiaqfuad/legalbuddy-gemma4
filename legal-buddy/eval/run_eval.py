"""Run retrieval eval for one collection against the gold set.

For each gold question: embed (query: prefix), search top-N chunks, then score two
granularities:
  - chunk-level : gold section among the sections of the first k RAW chunks
                  (this is what the baseline pipeline returns today).
  - section-level: gold among the first k UNIQUE sections after dedupe
                  (this is what the improved parent-document retrieval returns).
Reports recall@{1,3,5,10} and MRR for both. Writes eval/metrics_<tag>.json.

    PYTHONPATH=eval uv run python eval/run_eval.py --collection legal_acts_eval_baseline --tag baseline
"""
import argparse
import json

from common import (
    GOLDSET_PATH,
    build_qdrant_client,
    dedupe_preserve,
    embed_queries,
    metrics_path,
    recall_at_k,
    reciprocal_rank,
)

# Candidate chunks pulled per query before scoring. Must be large enough that, after
# collapsing chunk parts to unique sections, >=10 distinct sections remain — the
# improved chunker emits many small parts per section, so a small pool would starve
# section-level recall@10 (a harness artifact, not a chunking difference).
TOP_N = 100
KS = (1, 3, 5, 10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    gold = json.loads(GOLDSET_PATH.read_text(encoding="utf-8"))
    client = build_qdrant_client()
    vectors = embed_queries([g["question"] for g in gold])

    per_q = []
    agg = {
        "chunk": {"recall": {k: 0.0 for k in KS}, "mrr": 0.0},
        "section": {"recall": {k: 0.0 for k in KS}, "mrr": 0.0},
    }
    for g, vec in zip(gold, vectors):
        hits = client.query_points(
            collection_name=args.collection, query=vec, limit=TOP_N, with_payload=True
        ).points
        chunk_keys = [str((h.payload or {}).get("gold_key")) for h in hits]
        section_keys = dedupe_preserve(chunk_keys)
        target = g["gold_key"]

        c_rec = recall_at_k(chunk_keys, target, KS)
        s_rec = recall_at_k(section_keys, target, KS)
        c_mrr = reciprocal_rank(chunk_keys, target)
        s_mrr = reciprocal_rank(section_keys, target)

        for k in KS:
            agg["chunk"]["recall"][k] += c_rec[k]
            agg["section"]["recall"][k] += s_rec[k]
        agg["chunk"]["mrr"] += c_mrr
        agg["section"]["mrr"] += s_mrr
        per_q.append(
            {
                "question": g["question"],
                "gold_key": target,
                "found_section_rank": (
                    section_keys.index(target) + 1 if target in section_keys else None
                ),
                "hit@5_section": bool(s_rec[5]),
            }
        )

    n = len(gold)
    for gran in ("chunk", "section"):
        for k in KS:
            agg[gran]["recall"][k] = round(agg[gran]["recall"][k] / n, 4)
        agg[gran]["mrr"] = round(agg[gran]["mrr"] / n, 4)

    out = {"tag": args.tag, "collection": args.collection, "n": n, "metrics": agg, "per_q": per_q}
    path = metrics_path(args.tag)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"=== {args.tag} ({args.collection}) n={n} ===")
    for gran in ("chunk", "section"):
        r = agg[gran]["recall"]
        print(
            f"  {gran:8s} recall@1={r[1]:.3f} @3={r[3]:.3f} @5={r[5]:.3f} @10={r[10]:.3f} "
            f"MRR={agg[gran]['mrr']:.3f}"
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
