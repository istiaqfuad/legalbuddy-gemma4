import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=20)
    except Exception as exc:
        return f"ERROR: {exc}"


root_candidates = ["/marimo", "/home/oai", "/kaggle/working", "/workspace", "/tmp"]
paths = {}
for candidate in root_candidates:
    p = Path(candidate)
    if p.exists():
        try:
            paths[candidate] = sorted(str(x) for x in p.iterdir())[:80]
        except Exception as exc:
            paths[candidate] = [f"ERROR: {exc}"]

packages = {}
for name in ["torch", "transformers", "datasets", "trl", "peft", "unsloth", "accelerate"]:
    try:
        mod = __import__(name)
        packages[name] = getattr(mod, "__version__", "installed")
    except Exception as exc:
        packages[name] = f"missing: {exc}"

info = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "cwd": os.getcwd(),
    "disk": shutil.disk_usage("/"),
    "nvidia_smi": _run(["nvidia-smi"]),
    "packages": packages,
    "paths": paths,
}

print(json.dumps(info, default=str, ensure_ascii=False, indent=2))
