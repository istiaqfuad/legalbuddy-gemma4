import json
import zipfile
from pathlib import Path

zip_path = Path("/marimo/training/lawbuddy_27b_16bit_results.zip")
out = {"exists": zip_path.exists(), "entries": []}
if zip_path.exists():
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist()[:200]:
            out["entries"].append({"name": info.filename, "size_mb": round(info.file_size / 1024 / 1024, 3)})
print(json.dumps(out, indent=2))
