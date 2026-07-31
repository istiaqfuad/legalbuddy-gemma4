"""3-way compare: baseline (128, char-chunks) vs improved-128 vs improved-512.
Reads metrics_baseline.json, metrics_improved.json, metrics_improved512.json.
Uses section-level recall (parent-document) as the effective metric."""
import json

from common import metrics_path

KS = ("1", "3", "5", "10")
TAGS = [
    ("baseline", "chunk"),
    ("improved", "section"),
    ("improved512", "section"),
    ("mle5base512", "section"),
]


def load(tag):
    p = metrics_path(tag)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main():
    rows = []
    for tag, gran in TAGS:
        m = load(tag)
        if not m:
            print(f"(missing metrics_{tag}.json)")
            continue
        rec = {str(k): v for k, v in m["metrics"][gran]["recall"].items()}
        rows.append((tag, gran, rec, m["metrics"][gran]["mrr"], m["n"]))

    n = rows[0][4] if rows else 0
    print(f"n questions: {n}\n")
    hdr = f"{'variant':22s} {'gran':8s}  R@1    R@3    R@5    R@10   MRR"
    print(hdr)
    print("-" * len(hdr))
    for tag, gran, rec, mrr, _ in rows:
        print(
            f"{tag:22s} {gran:8s}  {rec['1']:.3f}  {rec['3']:.3f}  {rec['5']:.3f}  "
            f"{rec['10']:.3f}  {mrr:.3f}"
        )

    base = next((r for r in rows if r[0] == "baseline"), None)
    if base:
        print("\nΔ vs baseline (section/chunk effective):")
        b = base[2]
        for tag, gran, rec, mrr, _ in rows:
            if tag == "baseline":
                continue
            d = " ".join(f"@{k}={rec[k]-b[k]:+.3f}" for k in KS)
            print(f"  {tag:22s} {d}")


if __name__ == "__main__":
    main()
