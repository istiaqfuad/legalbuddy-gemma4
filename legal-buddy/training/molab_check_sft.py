import csv
import json
from collections import Counter
from pathlib import Path

base = Path("/marimo/training/style_sft_prod")
manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
counts = Counter()
invalid = []
with (base / "validation.csv").open(encoding="utf-8") as f:
    for row in csv.DictReader(f):
        counts[(row["split"], row["kind"], row["valid"], row["reason"])] += 1
        if row["valid"] != "True":
            invalid.append(row)

print("MANIFEST")
print(json.dumps(manifest, ensure_ascii=False, indent=2))
print("VALIDATION_COUNTS")
for key, value in sorted(counts.items()):
    print(key, value)
print("INVALID_SAMPLE")
print(json.dumps(invalid[:10], ensure_ascii=False, indent=2))

print("SAMPLES")
for path in [base / "train.jsonl", base / "eval.jsonl"]:
    print("==", path.name, "==")
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()[:3], 1):
        obj = json.loads(line)
        print(f"--- {path.name} row {i} ---")
        print(json.dumps(obj, ensure_ascii=False, indent=2)[:3500])
