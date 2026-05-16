"""
Append solved synthetic JSONL (from solve_puzzles_ollama.py or solve_puzzles_openrouter.py) into train_split_with_cot.csv.

Expected JSONL rows include at least: id, prompt, answer, type, generated_cot.
Extra keys (expected_answer, solved_by, …) are ignored.

Typical pipeline:
  python generate_puzzles.py
  python solve_puzzles_ollama.py --from-generated -o generated_puzzles_solved.jsonl
  python convert_extra_cots.py --in-place --backup
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

_DIR = Path(__file__).resolve().parent
_DEFAULT_TRAIN = _DIR / "train_split_with_cot.csv"
_DEFAULT_JSONL = _DIR / "generated_puzzles_solved.jsonl"
_DEFAULT_MERGED = _DIR / "train_split_with_cot_merged.csv"
_COLS = ["id", "prompt", "answer", "type", "generated_cot"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--train-csv",
        type=Path,
        default=_DEFAULT_TRAIN,
        help="Base CoT training CSV (default: train_split_with_cot.csv next to this script)",
    )
    p.add_argument(
        "--jsonl",
        type=Path,
        default=_DEFAULT_JSONL,
        help="Solved JSONL from solvers (default: generated_puzzles_solved.jsonl)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=f"When not using --in-place: full merged CSV path (default: {_DEFAULT_MERGED.name})",
    )
    p.add_argument(
        "--in-place",
        action="store_true",
        help="Write merged result back to --train-csv (same path as input base).",
    )
    p.add_argument(
        "--backup",
        action="store_true",
        help="With --in-place, copy the original train CSV to train_split_with_cot.csv.bak first.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts only; do not write files.",
    )
    args = p.parse_args()
    if args.in_place and args.output is not None:
        p.error("Do not pass --output with --in-place (output is --train-csv).")
    if not args.in_place and args.output is None:
        args.output = _DEFAULT_MERGED
    return args


def main() -> None:
    args = parse_args()
    train_path = args.train_csv.resolve()
    jsonl_path = args.jsonl.resolve()

    if not train_path.is_file():
        print(f"❌ Train CSV not found: {train_path}", file=sys.stderr)
        sys.exit(1)
    if not jsonl_path.is_file():
        print(f"❌ JSONL not found: {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    df_train = pd.read_csv(train_path)
    missing = [c for c in _COLS if c not in df_train.columns]
    if missing:
        print(f"❌ Train CSV missing columns {missing}", file=sys.stderr)
        sys.exit(1)

    existing_ids = set(df_train["id"].astype(str).tolist())
    new_rows: list[dict] = []
    skipped_dup = 0
    skipped_bad = 0

    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            skipped_bad += 1
            continue

        rid = str(data.get("id", "")).strip()
        answer = data.get("answer")
        cot = data.get("generated_cot")
        prompt = data.get("prompt")
        ptype = data.get("type")

        if not rid or prompt is None or answer is None or cot is None:
            skipped_bad += 1
            continue
        if str(cot).strip() == "":
            skipped_bad += 1
            continue

        if rid in existing_ids:
            skipped_dup += 1
            continue

        new_rows.append(
            {
                "id": rid,
                "prompt": prompt,
                "answer": str(answer).strip(),
                "type": ptype if ptype is not None and str(ptype).strip() else "Unknown",
                "generated_cot": cot,
            }
        )
        existing_ids.add(rid)

    print(f"Train rows: {len(df_train)}")
    print(f"New rows to append: {len(new_rows)}")
    print(f"Skipped (id already in train): {skipped_dup}")
    print(f"Skipped (invalid / incomplete JSONL row): {skipped_bad}")

    if not new_rows:
        print("Nothing to write.")
        return

    df_new = pd.DataFrame(new_rows)[_COLS]
    df_out = pd.concat([df_train[_COLS], df_new], ignore_index=True)

    out_path = train_path if args.in_place else args.output.resolve()

    if args.dry_run:
        print(f"Dry run: would write {len(df_out)} total rows to {out_path}")
        return

    if args.in_place and args.backup:
        bak = train_path.with_suffix(train_path.suffix + ".bak")
        shutil.copy2(train_path, bak)
        print(f"Backup: {bak}")

    df_out.to_csv(out_path, index=False)
    print(f"Wrote: {out_path} ({len(df_out)} rows)")


if __name__ == "__main__":
    main()
