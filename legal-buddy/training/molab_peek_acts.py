import json
from pathlib import Path

for p in sorted(Path("/marimo/training/acts").glob("*.json"))[:5]:
    obj = json.loads(p.read_text(encoding="utf-8"))
    print("---", p.name, "---")
    print(json.dumps(obj, ensure_ascii=False, indent=2)[:3000])
