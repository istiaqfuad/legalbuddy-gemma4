import json
import os
from pathlib import Path


base = Path("/marimo/training")
rows = []
for p in sorted(base.rglob("*")):
    try:
        st = p.stat()
    except OSError:
        continue
    rel = str(p.relative_to(base))
    if len(rel.split(os.sep)) > 4:
        continue
    rows.append({
        "path": rel,
        "type": "dir" if p.is_dir() else "file",
        "size_mb": round(st.st_size / 1024 / 1024, 3) if p.is_file() else None,
    })

print(json.dumps(rows[:500], ensure_ascii=False, indent=2))
print(f"total_shown={min(len(rows), 500)} total_seen={len(rows)}")
