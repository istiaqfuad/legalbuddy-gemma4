import gzip
import json
from pathlib import Path


def count_jsonl(path: Path) -> int | None:
    if not path.exists():
        return None
    opener = gzip.open if path.suffix == ".gz" else open
    mode = "rt" if path.suffix == ".gz" else "r"
    with opener(path, mode, encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in f)


base = Path("/marimo/training")
targets = [
    "parsed_cases.jsonl",
    "parsed_cases.jsonl.gz",
    "parsed_cases_subset.jsonl",
    "acts.tar.gz",
    "style_sft/train.jsonl",
    "style_sft/eval.jsonl",
    "lawbuddy_gemma4_16bit_lora_prod.zip",
    "lawbuddy_27b_16bit_results.zip",
]

summary = {}
for rel in targets:
    p = base / rel
    summary[rel] = {
        "exists": p.exists(),
        "size_mb": round(p.stat().st_size / 1024 / 1024, 3) if p.exists() and p.is_file() else None,
        "jsonl_rows": count_jsonl(p) if p.exists() and (p.suffix == ".jsonl" or p.suffix == ".gz") else None,
    }

dirs = {}
for rel in ["acts", "cases_json", "style_sft", "outputs", "results", "adapter", "lawbuddy_adapter"]:
    p = base / rel
    dirs[rel] = {
        "exists": p.exists(),
        "files": sum(1 for x in p.rglob("*") if x.is_file()) if p.exists() else 0,
        "sample": [str(x.relative_to(p)) for x in sorted(p.rglob("*"))[:20]] if p.exists() else [],
    }

print(json.dumps({"files": summary, "dirs": dirs}, ensure_ascii=False, indent=2))
