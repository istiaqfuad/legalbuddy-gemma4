"""Pick a deterministic, representative subset of acts for the A/B eval.

Filters to acts that have sections and are not repealed, then takes an evenly
strided sample so size variety is preserved. The Penal Code (act-print-11) is
force-included. Writes eval/subset_acts.json (list of filename stems).
"""
import json

from common import ACTS_DIR, SUBSET_PATH

TARGET = 200
FORCE = {"act-print-11"}  # Penal Code, 1860


def main():
    candidates = []
    for fp in sorted(ACTS_DIR.glob("act-print-*.json")):
        try:
            obj = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if obj.get("csv_metadata", {}).get("is_repealed") is True:
            continue
        secs = obj.get("sections") or []
        if not secs:
            continue
        candidates.append((fp.stem, len(secs)))

    stems = [c[0] for c in candidates]
    n = len(stems)
    if n <= TARGET:
        chosen = stems
    else:
        step = n / TARGET
        chosen = [stems[int(i * step)] for i in range(TARGET)]
    chosen = sorted(set(chosen) | (FORCE & set(stems)))

    total_sections = sum(dict(candidates)[s] for s in chosen)
    SUBSET_PATH.write_text(json.dumps(chosen, indent=2), encoding="utf-8")
    print(f"candidates (with sections, not repealed): {n}")
    print(f"subset chosen: {len(chosen)} acts, ~{total_sections} sections")
    print(f"penal code included: {'act-print-11' in chosen}")


if __name__ == "__main__":
    main()
