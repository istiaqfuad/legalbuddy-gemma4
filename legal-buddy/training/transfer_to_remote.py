"""Transfer parsed_cases_subset.jsonl to the remote marimo notebook in chunks."""
import subprocess
import sys
from pathlib import Path

SKILL_DIR = Path("/home/istiaqfuad/Desktop/kaggle_hackathon/.agents/skills/marimo-pair")
SCRIPT = SKILL_DIR / "scripts" / "execute-code.sh"
URL = "https://sb-28f9631e43581e1d.sb.molab.run/"
SESSION = "s_e4w4p3"
TOKEN = "41b58c8bcdd712696e5f5f62c8b9c3cb3a5ebe8ef8a5741844bec4d08e8810a8"

B64_FILE = Path("/home/istiaqfuad/Desktop/kaggle_hackathon/legal-buddy/training/parsed_cases_b64.txt")
CHUNK_SIZE = 80000  # chars per chunk


def run_code(code: str, timeout: int = 60) -> str:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--url", URL, "--session", SESSION, "--token", TOKEN],
        input=code,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout + result.stderr


def main():
    data = B64_FILE.read_text()
    total = len(data)
    chunks = [data[i : i + CHUNK_SIZE] for i in range(0, total, CHUNK_SIZE)]
    print(f"Total base64: {total} chars, {len(chunks)} chunks")

    # Initialize file on remote
    print("Initializing remote file...")
    out = run_code(
        "import pathlib\n"
        "pathlib.Path('/marimo/training').mkdir(exist_ok=True)\n"
        "pathlib.Path('/marimo/training/b64_data.txt').write_text('')\n"
        "print('Initialized')\n"
    )
    print(f"  {out.strip().split(chr(10))[-1]}")

    # Send chunks
    for i, chunk in enumerate(chunks):
        print(f"  Sending chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)...")
        code = (
            f"_chunk = '''{chunk}'''\n"
            f"with open('/marimo/training/b64_data.txt', 'a') as _f:\n"
            f"    _f.write(_chunk)\n"
            f"print(f'Chunk {i + 1} written')\n"
        )
        out = run_code(code)
        last_line = out.strip().split("\n")[-1]
        print(f"    {last_line}")

    # Decode on remote
    print("Decoding on remote...")
    out = run_code(
        "import base64, gzip, pathlib\n"
        "_b64 = pathlib.Path('/marimo/training/b64_data.txt').read_text()\n"
        "_compressed = base64.b64decode(_b64)\n"
        "_data = gzip.decompress(_compressed)\n"
        "pathlib.Path('/marimo/training/parsed_cases.jsonl').write_bytes(_data)\n"
        "_size = len(_data)\n"
        "_lines = _data.count(b'\\n')\n"
        "print(f'Decoded: {_size} bytes, {_lines} lines')\n"
        "pathlib.Path('/marimo/training/b64_data.txt').unlink()\n"
        "print('Cleanup done')\n",
        timeout=30,
    )
    print(f"  {out.strip()}")
    print("Transfer complete!")


if __name__ == "__main__":
    main()
