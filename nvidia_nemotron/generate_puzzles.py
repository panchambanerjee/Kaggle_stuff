"""
SCRIPT 1: Generate puzzle prompts only.

All categories are generated in Python (no LLM). Prompts are validated before
being written so they match the train-set shape and internal consistency rules.
Each JSONL row includes ``expected_answer`` (ground truth) for the solver to
verify model \\boxed{} output without consensus.
"""

import json
import random
import string
import time
from pathlib import Path

from puzzle_quality import (
    CATEGORY_SPECS,
    load_category_prompt_index,
    load_jsonl_records,
    to_roman,
    validate_puzzle_text,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_JSONL = str(_SCRIPT_DIR / "generated_puzzles_unsolved.jsonl")
TRAIN_CSV = str(_SCRIPT_DIR / "train.csv")

NUM_PER_CATEGORY = 100
REFERENCE_PROMPTS_COUNT = 2
MAX_ATTEMPTS_MULTIPLIER = 8
DEBUG = True

TEXT_ARTICLES = ["the"]
TEXT_ADJECTIVES = [
    "ancient",
    "bright",
    "curious",
    "golden",
    "hidden",
    "magical",
    "mysterious",
    "silver",
    "strange",
    "wise",
]
TEXT_NOUNS = [
    "alice",
    "bird",
    "castle",
    "cat",
    "door",
    "dragon",
    "forest",
    "garden",
    "hatter",
    "island",
    "key",
    "knight",
    "map",
    "message",
    "mirror",
    "mouse",
    "palace",
    "potion",
    "princess",
    "puzzle",
    "queen",
    "rabbit",
    "river",
    "secret",
    "story",
    "student",
    "teacher",
    "treasure",
    "turtle",
    "valley",
    "village",
    "wizard",
]
TEXT_VERBS = [
    "builds",
    "chases",
    "creates",
    "discovers",
    "draws",
    "dreams",
    "explores",
    "follows",
    "finds",
    "hides",
    "imagines",
    "opens",
    "reads",
    "sees",
    "studies",
    "watches",
    "writes",
]
TEXT_PREPOSITIONS = [
    "above",
    "after",
    "behind",
    "in",
    "inside",
    "near",
    "under",
]

EQUATION_PUNCT_SOURCE_ALPHABET = list(r"""!@#$%^&*()_+-[]{}|:;'"<>,.?/`\~""")
EQUATION_DIGIT_SOURCE_ALPHABET = list("0123456789") + list(r"""/\|{}<>+-*`'"#@$%^&!?""")
EQUATION_RESULT_ALPHABET = list("0123456789") + list(r"""/\|{}<>+-*`'\"#@$%^&!?[]():;,.%""")


def load_existing_prompts_by_category():
    prompts_by_category = {name: [] for name in CATEGORY_SPECS}
    for record in load_jsonl_records(OUTPUT_JSONL):
        category_name = record.get("type")
        prompt = record.get("prompt")
        if category_name in prompts_by_category and prompt:
            prompts_by_category[category_name].append(prompt)
    return prompts_by_category


def _extract_hidden_gravity(example_prompt):
    lines = [line.strip() for line in example_prompt.splitlines() if line.strip()]
    observations = []
    for line in lines:
        match = CATEGORY_SPECS["Gravitational Constant"]["example_pattern"].fullmatch(line)
        if match:
            observations.append((float(match.group(1)), float(match.group(2))))

    if not observations:
        return random.uniform(5.0, 15.0)

    inferred_g_values = []
    for time_s, distance_m in observations:
        inferred_g_values.append((2.0 * distance_m) / (time_s * time_s))
    return sum(inferred_g_values) / len(inferred_g_values)


def _format_decimal(value):
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _extract_hidden_unit_ratio(example_prompt):
    pattern = CATEGORY_SPECS["Unit Conversion"]["example_pattern"]
    ratios = []
    for line in example_prompt.splitlines():
        line = line.strip()
        match = pattern.fullmatch(line)
        if match:
            measurement = float(match.group(1))
            converted = float(match.group(2))
            if measurement > 0:
                ratios.append(converted / measurement)
    if ratios:
        return sum(ratios) / len(ratios)
    return random.uniform(0.65, 1.45)


def generate_unit_conversion_puzzle(example_prompts):
    """One hidden multiplicative factor; examples and query consistent after rounding."""
    spec = CATEGORY_SPECS["Unit Conversion"]
    example_count = random.randint(spec["min_examples"], spec["max_examples"])
    template = random.choice(example_prompts)
    base_ratio = _extract_hidden_unit_ratio(template)

    pairs = None
    query_measurement = None

    for _ in range(160):
        ratio = max(0.35, min(2.25, base_ratio * random.uniform(0.94, 1.06)))
        pool = [round(value / 100.0, 2) for value in range(600, 6000)]
        random.shuffle(pool)
        chosen = []
        for measurement in pool:
            if measurement <= 0:
                continue
            converted = round(measurement * ratio, 2)
            if converted <= 0:
                continue
            chosen.append((measurement, converted))
            if len(chosen) == example_count + 1:
                break
        if len(chosen) < example_count + 1:
            continue

        ratios = [converted / measurement for measurement, converted in chosen]
        if max(ratios) - min(ratios) > 0.018:
            continue

        pairs = chosen[:-1]
        query_measurement = chosen[-1][0]
        break

    if pairs is None:
        ratio = 0.75
        base_measurements = [10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0][: example_count + 1]
        chosen = [(m, round(m * ratio, 2)) for m in base_measurements]
        pairs = chosen[:-1]
        query_measurement = chosen[-1][0]

    lines = [
        spec["intro"],
        "",
        spec["section_header"],
    ]
    for measurement, converted in pairs:
        lines.append(f"{_format_decimal(measurement)} m becomes {_format_decimal(converted)}")

    lines.extend(
        [
            "",
            f"Now, convert the following measurement: {_format_decimal(query_measurement)} m",
        ]
    )
    expected_answer = _format_decimal(chosen[-1][1])
    return "\n".join(lines), expected_answer


def _bit_reverse8(value):
    value &= 0xFF
    bits = format(value, "08b")
    return int(bits[::-1], 2)


def _rol8(value, steps):
    value &= 0xFF
    steps %= 8
    return ((value << steps) | (value >> (8 - steps))) & 0xFF


def _random_bit_chain():
    xor_mask = random.randint(1, 255)
    rotation = random.randint(1, 7)
    pieces = [
        lambda x, m=xor_mask: (x ^ m) & 0xFF,
        lambda x: _bit_reverse8(x),
        lambda x, r=rotation: _rol8(x, r),
        lambda x: ((x & 0x0F) << 4) | ((x & 0xF0) >> 4),
        lambda x: (~x) & 0xFF,
    ]
    random.shuffle(pieces)
    count = random.randint(2, min(4, len(pieces)))
    return pieces[:count]


def _apply_bit_chain(value, chain):
    output = value & 0xFF
    for step in chain:
        output = step(output) & 0xFF
    return output


def generate_bit_manipulation_puzzle():
    """Composable bijections on 8-bit strings; format matches train."""
    spec = CATEGORY_SPECS["Bit Manipulation"]
    example_count = random.randint(spec["min_examples"], spec["max_examples"])

    for _ in range(200):
        chain = _random_bit_chain()
        inputs = random.sample(range(256), example_count + 1)
        outputs = [_apply_bit_chain(value, chain) for value in inputs]

        if len(set(outputs[:example_count])) == 1:
            continue

        example_inputs = inputs[:example_count]
        query_input = inputs[example_count]
        if query_input in set(example_inputs):
            continue

        lines = [
            spec["intro"],
            "",
            spec["section_header"],
        ]
        for value in example_inputs:
            lines.append(f"{format(value, '08b')} -> {format(_apply_bit_chain(value, chain), '08b')}")

        lines.extend(["", f"Now, determine the output for: {format(query_input, '08b')}"])
        expected_answer = format(_apply_bit_chain(query_input, chain), "08b")
        return "\n".join(lines), expected_answer

    mask = random.randint(1, 255)
    chain = [lambda x, m=mask: (x ^ m) & 0xFF]
    inputs = random.sample(range(256), example_count + 1)
    example_inputs = inputs[:example_count]
    query_input = inputs[example_count]
    lines = [
        spec["intro"],
        "",
        spec["section_header"],
    ]
    for value in example_inputs:
        lines.append(f"{format(value, '08b')} -> {format(_apply_bit_chain(value, chain), '08b')}")
    lines.extend(["", f"Now, determine the output for: {format(query_input, '08b')}"])
    expected_answer = format(_apply_bit_chain(query_input, chain), "08b")
    return "\n".join(lines), expected_answer


def generate_gravity_puzzle(example_prompts):
    """Generate gravity prompts deterministically so they always satisfy one hidden g."""
    template_prompt = random.choice(example_prompts)
    hidden_g = _extract_hidden_gravity(template_prompt)

    example_count = random.randint(
        CATEGORY_SPECS["Gravitational Constant"]["min_examples"],
        CATEGORY_SPECS["Gravitational Constant"]["max_examples"],
    )

    candidate_times = [round(value / 100.0, 2) for value in range(100, 496)]
    sampled_times = random.sample(candidate_times, example_count + 1)
    example_times = sampled_times[:-1]
    query_time = sampled_times[-1]

    lines = [
        CATEGORY_SPECS["Gravitational Constant"]["intro"],
        "",
        CATEGORY_SPECS["Gravitational Constant"]["section_header"],
    ]

    for time_s in example_times:
        distance_m = 0.5 * hidden_g * time_s * time_s
        lines.append(
            f"For t = {_format_decimal(time_s)}s, distance = {_format_decimal(distance_m)} m"
        )

    lines.extend(
        [
            "",
            (
                "Now, determine the falling distance for t = "
                f"{_format_decimal(query_time)}s given d = 0.5*g*t^2."
            ),
        ]
    )
    expected_answer = _format_decimal(0.5 * hidden_g * query_time * query_time)
    return "\n".join(lines), expected_answer


def _random_substitution_cipher():
    letters = list(string.ascii_lowercase)
    shuffled = letters[:]
    random.shuffle(shuffled)
    return dict(zip(letters, shuffled))


def _encrypt_with_cipher(plaintext, cipher_map):
    return "".join(cipher_map.get(char, char) for char in plaintext)


def _build_plaintext_phrase():
    templates = [
        [random.choice(TEXT_NOUNS), random.choice(TEXT_VERBS), random.choice(TEXT_NOUNS)],
        [
            random.choice(TEXT_NOUNS),
            random.choice(TEXT_VERBS),
            random.choice(TEXT_PREPOSITIONS),
            random.choice(TEXT_NOUNS),
        ],
        [
            random.choice(TEXT_ARTICLES),
            random.choice(TEXT_ADJECTIVES),
            random.choice(TEXT_NOUNS),
            random.choice(TEXT_VERBS),
        ],
        [
            random.choice(TEXT_ARTICLES),
            random.choice(TEXT_ADJECTIVES),
            random.choice(TEXT_NOUNS),
            random.choice(TEXT_VERBS),
            random.choice(TEXT_PREPOSITIONS),
            random.choice(TEXT_NOUNS),
        ],
        [
            random.choice(TEXT_NOUNS),
            random.choice(TEXT_VERBS),
            random.choice(TEXT_ARTICLES),
            random.choice(TEXT_ADJECTIVES),
            random.choice(TEXT_NOUNS),
        ],
    ]
    return " ".join(random.choice(templates))


def generate_text_encryption_puzzle():
    """Generate substitution-cipher prompts deterministically."""
    example_count = random.randint(
        CATEGORY_SPECS["Text Encryption"]["min_examples"],
        CATEGORY_SPECS["Text Encryption"]["max_examples"],
    )
    cipher_map = _random_substitution_cipher()

    plaintext_phrases = set()
    while len(plaintext_phrases) < example_count + 1:
        plaintext_phrases.add(_build_plaintext_phrase())
    plaintext_phrases = list(plaintext_phrases)
    random.shuffle(plaintext_phrases)

    example_plaintexts = plaintext_phrases[:-1]
    query_plaintext = plaintext_phrases[-1]

    lines = [
        CATEGORY_SPECS["Text Encryption"]["intro"],
        "",
        CATEGORY_SPECS["Text Encryption"]["section_header"],
    ]

    for plaintext in example_plaintexts:
        ciphertext = _encrypt_with_cipher(plaintext, cipher_map)
        lines.append(f"{ciphertext} -> {plaintext}")

    lines.extend(
        [
            "",
            (
                "Now, decrypt the following text: "
                f"{_encrypt_with_cipher(query_plaintext, cipher_map)}"
            ),
        ]
    )
    return "\n".join(lines), query_plaintext


def generate_numeral_conversion_puzzle():
    """Generate Roman-numeral prompts deterministically."""
    example_count = random.randint(
        CATEGORY_SPECS["Numeral Conversion"]["min_examples"],
        CATEGORY_SPECS["Numeral Conversion"]["max_examples"],
    )
    sampled_numbers = random.sample(range(1, 101), example_count + 1)
    example_numbers = sampled_numbers[:-1]
    query_number = sampled_numbers[-1]

    lines = [
        CATEGORY_SPECS["Numeral Conversion"]["intro"],
        "",
        CATEGORY_SPECS["Numeral Conversion"]["section_header"],
    ]

    for number in example_numbers:
        lines.append(f"{number} -> {to_roman(number)}")

    lines.extend(
        [
            "",
            (
                "Now, write the number "
                f"{query_number} in the Wonderland numeral system."
            ),
        ]
    )
    return "\n".join(lines), to_roman(query_number)


def _random_equation_token(alphabet, min_length=4, max_length=6):
    length = random.randint(min_length, max_length)
    return "".join(random.choice(alphabet) for _ in range(length))


def generate_equation_transformation_puzzle():
    """Generate equation prompts deterministically."""
    example_count = random.randint(
        CATEGORY_SPECS["Equation Transformation"]["min_examples"],
        CATEGORY_SPECS["Equation Transformation"]["max_examples"],
    )
    use_digits = random.random() < 0.47
    source_alphabet = (
        EQUATION_DIGIT_SOURCE_ALPHABET if use_digits else EQUATION_PUNCT_SOURCE_ALPHABET
    )

    selected_chars = random.sample(
        source_alphabet,
        k=min(len(source_alphabet), random.randint(8, 14)),
    )
    target_chars = random.sample(EQUATION_RESULT_ALPHABET, k=len(selected_chars))
    transform_map = dict(zip(selected_chars, target_chars))

    examples = []
    seen_sources = set()

    while len(examples) < example_count + 1:
        source = _random_equation_token(selected_chars)
        if source in seen_sources:
            continue
        target = "".join(transform_map[char] for char in source)
        seen_sources.add(source)
        examples.append((source, target))

    example_pairs = examples[:-1]
    query_token = examples[-1][0]
    expected_answer = examples[-1][1]

    lines = [
        CATEGORY_SPECS["Equation Transformation"]["intro"],
        "",
        CATEGORY_SPECS["Equation Transformation"]["section_header"],
    ]
    for source, target in example_pairs:
        lines.append(f"{source} = {target}")

    lines.extend(
        [
            "",
            f"Now, determine the result for: {query_token}",
        ]
    )
    return "\n".join(lines), expected_answer


def generate_puzzle_text_only(category_name, example_prompts):
    """Generate (prompt, expected_answer) for one category (all paths are Python-only)."""
    if category_name == "Bit Manipulation":
        return generate_bit_manipulation_puzzle()
    if category_name == "Unit Conversion":
        return generate_unit_conversion_puzzle(example_prompts)
    if category_name == "Gravitational Constant":
        return generate_gravity_puzzle(example_prompts)
    if category_name == "Text Encryption":
        return generate_text_encryption_puzzle()
    if category_name == "Numeral Conversion":
        return generate_numeral_conversion_puzzle()
    if category_name == "Equation Transformation":
        return generate_equation_transformation_puzzle()
    raise ValueError(f"Unknown category: {category_name}")


def save_puzzle_record(record):
    with open(OUTPUT_JSONL, "a", encoding="utf-8") as handle:
        json.dump(record, handle)
        handle.write("\n")


def generator_label_for_category(category_name):
    if category_name == "Bit Manipulation":
        return "python-deterministic-bits"
    if category_name == "Unit Conversion":
        return "python-deterministic-units"
    if category_name == "Gravitational Constant":
        return "python-deterministic-gravity"
    if category_name == "Text Encryption":
        return "python-deterministic-substitution"
    if category_name == "Numeral Conversion":
        return "python-deterministic-roman"
    if category_name == "Equation Transformation":
        return "python-deterministic-equations"
    return "python-deterministic"


def main():
    print("=" * 70)
    print("PUZZLE GENERATOR (Questions Only)")
    print("=" * 70)
    print("Mode: Python-only (no LLM)")
    print(f"Output: {OUTPUT_JSONL}")
    print()

    print(f"Loading {TRAIN_CSV}...")
    try:
        train_prompts_by_category = load_category_prompt_index(TRAIN_CSV)
    except FileNotFoundError:
        print(f"❌ {TRAIN_CSV} not found!")
        return

    total_train_examples = sum(len(prompts) for prompts in train_prompts_by_category.values())
    print(f"✓ Loaded {total_train_examples} prompts across categories\n")

    existing_prompts_by_category = load_existing_prompts_by_category()
    total_generated = 0

    for category_name in CATEGORY_SPECS:
        category_prompts = train_prompts_by_category[category_name]
        seen_prompts = list(existing_prompts_by_category[category_name])
        existing_count = len(seen_prompts)
        remaining_target = max(0, NUM_PER_CATEGORY - existing_count)

        print(f"\n{'=' * 70}")
        print(category_name)
        print(f"{'=' * 70}")

        if not category_prompts:
            print("⚠️ No train examples found, skipping")
            continue

        print(f"Train examples: {len(category_prompts)}")
        print(f"Already saved: {existing_count}")
        print(f"Target this run: {remaining_target}\n")

        if remaining_target == 0:
            print("Category already reached target count, skipping.")
            continue

        successes = 0
        attempts = 0
        max_attempts = remaining_target * MAX_ATTEMPTS_MULTIPLIER

        while successes < remaining_target and attempts < max_attempts:
            attempts += 1
            print(f"[{successes + 1}/{remaining_target}] Attempt {attempts}...", end=" ")

            reference_prompts = random.sample(
                category_prompts,
                k=min(REFERENCE_PROMPTS_COUNT, len(category_prompts)),
            )
            try:
                puzzle_text, expected_answer = generate_puzzle_text_only(
                    category_name, reference_prompts
                )
            except ValueError as exc:
                print(f"  ✗ {exc}")
                print("❌ FAILED")
                continue

            if not puzzle_text:
                print("❌ FAILED")
                continue

            issues = validate_puzzle_text(
                puzzle_text,
                category_name,
                reference_prompts=category_prompts,
                seen_prompts=seen_prompts,
            )
            if issues:
                print(f"  ✗ {issues[0]}")
                if DEBUG and len(issues) > 1:
                    print(f"    More: {', '.join(issues[1:3])}")
                print("❌ FAILED")
                continue

            puzzle_record = {
                "id": (
                    f"gen_{category_name[:3].replace(' ', '')}_"
                    f"{int(time.time())}_{existing_count + successes}"
                ),
                "prompt": puzzle_text,
                "type": category_name,
                "expected_answer": expected_answer,
                "generated_by": generator_label_for_category(category_name),
                "temperature": CATEGORY_SPECS[category_name]["temperature"],
                "status": "unsolved",
            }
            save_puzzle_record(puzzle_record)

            seen_prompts.append(puzzle_text)
            successes += 1
            total_generated += 1

            print("✅ SUCCESS")
            if DEBUG:
                print(f"   Length: {len(puzzle_text)} chars")
                print(f"   Preview: {puzzle_text[:100]}...")

        print(f"\nCategory complete: {existing_count + successes}/{NUM_PER_CATEGORY}")
        if attempts >= max_attempts and successes < remaining_target:
            print("Stopped after hitting the attempt limit.")

    print(f"\n{'=' * 70}")
    print("GENERATION COMPLETE!")
    print(f"{'=' * 70}")
    print(f"Generated this run: {total_generated} puzzle questions")
    print(f"Saved to: {OUTPUT_JSONL}")
    print()
    print("Next steps:")
    print("  1. Run: python filter_generated_puzzles.py")
    print("  2. Review accepted / rejected outputs")
    print("  3. Run: python solve_puzzles_ollama.py --from-generated -o generated_puzzles_solved.jsonl")
    print("  4. Append to train_split_with_cot.csv: python convert_extra_cots.py --in-place --backup")
    print(f"{'=' * 70}")


def test_mode():
    """Quick smoke test (Python-only)."""
    print("=" * 70)
    print("TEST MODE")
    print("=" * 70)
    print()

    train_prompts_by_category = load_category_prompt_index(TRAIN_CSV)

    for category_name in ("Bit Manipulation", "Unit Conversion"):
        reference_prompts = random.sample(
            train_prompts_by_category[category_name],
            k=min(REFERENCE_PROMPTS_COUNT, len(train_prompts_by_category[category_name])),
        )
        print(f"--- {category_name} ---\n")
        puzzle_text, expected_answer = generate_puzzle_text_only(
            category_name, reference_prompts
        )
        issues = validate_puzzle_text(
            puzzle_text,
            category_name,
            reference_prompts=train_prompts_by_category[category_name],
        )
        print(puzzle_text)
        print()
        print("EXPECTED:", repr(expected_answer))
        print("ISSUES:", issues if issues else "(none)")
        print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        DEBUG = True
        test_mode()
    else:
        main()