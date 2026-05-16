import json
from collections import Counter, defaultdict
from pathlib import Path

from puzzle_quality import (
    CATEGORY_SPECS,
    load_category_prompt_index,
    load_jsonl_records,
    validate_puzzle_text,
)


_SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_CSV = str(_SCRIPT_DIR / "train.csv")
INPUT_JSONL = str(_SCRIPT_DIR / "generated_puzzles_unsolved.jsonl")
OUTPUT_JSONL = str(_SCRIPT_DIR / "generated_puzzles_filtered.jsonl")
REJECTED_JSONL = str(_SCRIPT_DIR / "generated_puzzles_rejected.jsonl")
REPORT_JSON = str(_SCRIPT_DIR / "generated_puzzles_filter_report.json")


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle)
            handle.write("\n")


def main():
    print("=" * 70)
    print("FILTER GENERATED PUZZLES")
    print("=" * 70)
    print(f"Input: {INPUT_JSONL}")
    print(f"Accepted output: {OUTPUT_JSONL}")
    print(f"Rejected output: {REJECTED_JSONL}")
    print()

    train_prompts_by_category = load_category_prompt_index(TRAIN_CSV)
    generated_records = load_jsonl_records(INPUT_JSONL)

    if not generated_records:
        print("No generated puzzles found to filter.")
        return

    accepted_records = []
    rejected_records = []
    seen_prompts_by_category = defaultdict(list)
    rejection_reasons = Counter()
    accepted_by_category = Counter()
    rejected_by_category = Counter()

    for record in generated_records:
        category_name = record.get("type")
        prompt = record.get("prompt", "")

        if category_name not in CATEGORY_SPECS:
            record_with_reason = {
                **record,
                "rejection_reasons": ["unknown category"],
            }
            rejected_records.append(record_with_reason)
            rejection_reasons["unknown category"] += 1
            continue

        issues = validate_puzzle_text(
            prompt,
            category_name,
            reference_prompts=train_prompts_by_category[category_name],
            seen_prompts=seen_prompts_by_category[category_name],
        )

        if issues:
            record_with_reason = {**record, "rejection_reasons": issues}
            rejected_records.append(record_with_reason)
            rejected_by_category[category_name] += 1
            for issue in issues:
                rejection_reasons[issue] += 1
            continue

        accepted_records.append(record)
        accepted_by_category[category_name] += 1
        seen_prompts_by_category[category_name].append(prompt)

    write_jsonl(OUTPUT_JSONL, accepted_records)
    write_jsonl(REJECTED_JSONL, rejected_records)

    report = {
        "input_count": len(generated_records),
        "accepted_count": len(accepted_records),
        "rejected_count": len(rejected_records),
        "accepted_by_category": dict(sorted(accepted_by_category.items())),
        "rejected_by_category": dict(sorted(rejected_by_category.items())),
        "top_rejection_reasons": rejection_reasons.most_common(20),
    }

    with open(REPORT_JSON, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Accepted: {len(accepted_records)}")
    print(f"Rejected: {len(rejected_records)}")
    print()
    print("Accepted by category:")
    for category_name in CATEGORY_SPECS:
        print(f"  {category_name}: {accepted_by_category[category_name]}")

    if rejection_reasons:
        print("\nTop rejection reasons:")
        for reason, count in rejection_reasons.most_common(10):
            print(f"  {count:>3}  {reason}")

    print(f"\nSaved report to: {REPORT_JSON}")
    print("=" * 70)


if __name__ == "__main__":
    main()
