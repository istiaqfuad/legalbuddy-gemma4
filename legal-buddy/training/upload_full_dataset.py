"""Transfer full compressed dataset to the remote marimo notebook."""
import base64
import json
import urllib.request
from pathlib import Path

URL = "https://sb-28f9631e43581e1d.sb.molab.run/"
SESSION = "s_e4w4p3"
TOKEN = "41b58c8bcdd712696e5f5f62c8b9c3cb3a5ebe8ef8a5741844bec4d08e8810a8"

FILES_TO_UPLOAD = {
    "/marimo/training/acts.tar.gz": Path("/home/istiaqfuad/Desktop/kaggle_hackathon/legal-buddy/training/acts.tar.gz"),
    "/marimo/training/parsed_cases.jsonl.gz": Path("/home/istiaqfuad/Desktop/kaggle_hackathon/legal-buddy/training/parsed_cases.jsonl.gz")
}

CHUNK_SIZE = 80000  # 80KB base64 chars per chunk

SKILL_DIR = Path("/home/istiaqfuad/Desktop/kaggle_hackathon/.agents/skills/marimo-pair")
SCRIPT = SKILL_DIR / "scripts" / "execute-code.sh"

def execute_marimo_code(code: str) -> None:
    import subprocess
    result = subprocess.run(
        ["bash", str(SCRIPT), "--url", URL, "--token", TOKEN],
        input=code,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise Exception(f"Failed to execute code: {result.stderr}")

def main():
    print("Initializing remote directories...")
    execute_marimo_code(
        "import pathlib\n"
        "pathlib.Path('/marimo/training').mkdir(exist_ok=True)\n"
    )

    for remote_path, local_path in FILES_TO_UPLOAD.items():
        if not local_path.exists():
            print(f"Skipping {local_path} (not found)")
            continue
            
        print(f"\\nReading {local_path} ({local_path.stat().st_size / 1024 / 1024:.1f} MB)...")
        data = local_path.read_bytes()
        encoded = base64.b64encode(data).decode('utf-8')
        total = len(encoded)
        chunks = [encoded[i:i+CHUNK_SIZE] for i in range(0, total, CHUNK_SIZE)]
        
        print(f"Uploading {local_path.name} in {len(chunks)} chunks...")
        
        # Clear remote file
        execute_marimo_code(f"import pathlib\npathlib.Path('{remote_path}.b64').write_text('')\n")
        
        # Upload chunks
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i+1}/{len(chunks)}...", end="\r")
            code = (
                f"_chunk = '''{chunk}'''\n"
                f"with open('{remote_path}.b64', 'a') as _f:\n"
                f"    _f.write(_chunk)\n"
            )
            execute_marimo_code(code)
            
        print(f"\\nDecoding {remote_path} on remote...")
        decode_code = (
            f"import base64, pathlib\n"
            f"_b64 = pathlib.Path('{remote_path}.b64').read_text()\n"
            f"pathlib.Path('{remote_path}').write_bytes(base64.b64decode(_b64))\n"
            f"pathlib.Path('{remote_path}.b64').unlink()\n"
        )
        execute_marimo_code(decode_code)
        print(f"Finished uploading {local_path.name}")
        
    print("\\nExtracting gzip files on remote...")
    extract_code = (
        "import gzip, pathlib, tarfile\n"
        # Decompress jsonl
        "if pathlib.Path('/marimo/training/parsed_cases.jsonl.gz').exists():\n"
        "    _data = gzip.decompress(pathlib.Path('/marimo/training/parsed_cases.jsonl.gz').read_bytes())\n"
        "    pathlib.Path('/marimo/training/parsed_cases.jsonl').write_bytes(_data)\n"
        # Extract tar
        "if pathlib.Path('/marimo/training/acts.tar.gz').exists():\n"
        "    with tarfile.open('/marimo/training/acts.tar.gz', 'r:gz') as tar:\n"
        "        tar.extractall('/marimo/training')\n"
    )
    execute_marimo_code(extract_code)
    print("Done!")

if __name__ == "__main__":
    main()
