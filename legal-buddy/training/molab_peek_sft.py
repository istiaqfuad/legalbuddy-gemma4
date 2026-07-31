import json
from pathlib import Path

path = Path("/marimo/training/style_sft/train.jsonl")
for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()[:3], 1):
    obj = json.loads(line)
    print(f"--- row {i} ---")
    print(json.dumps(obj, ensure_ascii=False, indent=2)[:5000])
