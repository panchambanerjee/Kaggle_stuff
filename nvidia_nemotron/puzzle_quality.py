import csv
import json
import re


def build_solver_prompt(puzzle_text: str) -> str:
    """
    User message framing for puzzle solving / CoT generation.
    Keep in sync with ``solve_puzzles_ollama.solve_puzzle_with_cot`` (and OpenRouter solver).
    """
    return (
        f"{puzzle_text}\n\n"
        "Solve this puzzle step by step. Deduce the secret rule from the examples, "
        "then apply it to answer the final question.\n\n"
        "You MUST end your response with exactly one final answer in \\boxed{...}. "
        "Do not finish without \\boxed{}. For example: \\boxed{your answer}\n"
        "For 8-bit binary outputs use eight digits inside the box, e.g. \\boxed{00101101}.\n\n"
        "Let's think step by step to deduce the secret rules."
    )


def extract_boxed_answer(text):
    """Extract the last \\boxed{...} value from model output."""
    if not text:
        return None
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    if matches:
        return matches[-1].strip()
    return None


def extract_model_answer(text: str, puzzle_type: str) -> str | None:
    """Prefer \\boxed{}; light fallbacks when models ignore the format."""
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed
    if not text:
        return None

    if puzzle_type == "Bit Manipulation":
        bits = re.findall(r"\b[01]{8}\b", text)
        if bits:
            return bits[-1]

    if puzzle_type == "Numeral Conversion":
        romans = re.findall(r"\b[IVXLCDM]+\b", text.upper())
        if romans:
            return romans[-1]

    if puzzle_type in ("Gravitational Constant", "Unit Conversion"):
        nums = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
        if nums:
            return nums[-1]

    return None


CATEGORY_SPECS = {
    "Bit Manipulation": {
        "match_phrase": "bit manipulation rule transforms 8-bit binary numbers",
        "intro": (
            "In Alice's Wonderland, a secret bit manipulation rule transforms "
            "8-bit binary numbers. The transformation involves operations like "
            "bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority "
            "or choice functions."
        ),
        "section_header": "Here are some examples of input -> output:",
        "question_pattern": re.compile(
            r"^Now, determine the output for: ([01]{8})$"
        ),
        "example_pattern": re.compile(r"^([01]{8}) -> ([01]{8})$"),
        "min_examples": 6,
        "max_examples": 9,
        "max_similarity": None,
        "max_train_example_overlap": 4,
        "same_query_overlap_limit": 3,
        "banned_fragments": (
            "rule involves",
            "example 1:",
            "examples demonstrating",
            "first identifying",
            "flips the ",
        ),
        "temperature": 0.45,
    },
    "Gravitational Constant": {
        "match_phrase": "gravitational constant has been secretly changed",
        "intro": (
            "In Alice's Wonderland, the gravitational constant has been secretly "
            "changed."
        ),
        "section_header": "Here are some example observations:",
        "question_pattern": re.compile(
            r"^Now, determine the falling distance for t = "
            r"([0-9]+(?:\.[0-9]+)?)s given d = 0\.5\*g\*t\^2\.$"
        ),
        "example_pattern": re.compile(
            r"^For t = ([0-9]+(?:\.[0-9]+)?)s, distance = "
            r"([0-9]+(?:\.[0-9]+)?) m$"
        ),
        "min_examples": 4,
        "max_examples": 6,
        "max_similarity": None,
        "max_train_example_overlap": 3,
        "same_query_overlap_limit": 2,
        "banned_fragments": (
            "sin(",
            "time of day",
            "varies depending",
            "formula:",
            "k is a constant",
        ),
        "temperature": 0.35,
    },
    "Unit Conversion": {
        "match_phrase": "secret unit conversion is applied to measurements",
        "intro": (
            "In Alice's Wonderland, a secret unit conversion is applied to "
            "measurements."
        ),
        "section_header": "For example:",
        "question_pattern": re.compile(
            r"^Now, convert the following measurement: "
            r"([0-9]+(?:\.[0-9]+)?) m$"
        ),
        "example_pattern": re.compile(
            r"^([0-9]+(?:\.[0-9]+)?) m becomes ([0-9]+(?:\.[0-9]+)?)$"
        ),
        "min_examples": 3,
        "max_examples": 6,
        "max_similarity": None,
        "max_train_example_overlap": 3,
        "same_query_overlap_limit": 2,
        "banned_fragments": (
            "rule involves",
            "example 1:",
            "examples demonstrating",
        ),
        "temperature": 0.35,
    },
    "Text Encryption": {
        "match_phrase": "secret encryption rules are used on text",
        "intro": (
            "In Alice's Wonderland, secret encryption rules are used on text."
        ),
        "section_header": "Here are some examples:",
        "question_pattern": re.compile(
            r"^Now, decrypt the following text: ([a-z]+(?: [a-z]+){1,6})$"
        ),
        "example_pattern": re.compile(
            r"^([a-z]+(?: [a-z]+){1,6}) -> ([a-z]+(?: [a-z]+){1,6})$"
        ),
        "min_examples": 4,
        "max_examples": 6,
        "max_similarity": None,
        "max_train_example_overlap": 3,
        "same_query_overlap_limit": 2,
        "banned_fragments": (
            '"',
            "example 1:",
            "question:",
            "can you decipher",
            "can you decrypt",
            "the rule involves",
            "the magic here",
            "note:",
            "answer is not provided",
        ),
        "temperature": 0.35,
    },
    "Numeral Conversion": {
        "match_phrase": "numbers are secretly converted into a different numeral system",
        "intro": (
            "In Alice's Wonderland, numbers are secretly converted into a "
            "different numeral system."
        ),
        "section_header": "Some examples are given below:",
        "question_pattern": re.compile(
            r"^Now, write the number ([0-9]+) in the Wonderland numeral system\.$"
        ),
        "example_pattern": re.compile(r"^([0-9]+) -> ([IVXLCDM]+)$"),
        "min_examples": 4,
        "max_examples": 6,
        "max_similarity": None,
        "max_train_example_overlap": 4,
        "same_query_overlap_limit": 3,
        "banned_fragments": (
            "symbol",
            "emoji",
            "prime factor",
            "square root",
            "alphabet",
            "bees",
            "flowers",
            "star",
            "roman numerals",
        ),
        "temperature": 0.25,
    },
    "Equation Transformation": {
        "match_phrase": "secret set of transformation rules is applied to equations",
        "intro": (
            "In Alice's Wonderland, a secret set of transformation rules is "
            "applied to equations."
        ),
        "section_header": "Below are a few examples:",
        "question_pattern": re.compile(
            r"^Now, determine the result for: ([^A-Za-z\s=]{4,6})$"
        ),
        "example_pattern": re.compile(
            r"^([^A-Za-z\s=]{4,6}) = ([^A-Za-z\s=]{1,6})$"
        ),
        "min_examples": 4,
        "max_examples": 6,
        "max_similarity": None,
        "max_train_example_overlap": 3,
        "same_query_overlap_limit": 2,
        "banned_fragments": (
            "example 1:",
            "note:",
            "symbol represents",
            "operation that combines",
            "12 + 18",
        ),
        "temperature": 0.35,
    },
}


GENERIC_BANNED_FRAGMENTS = (
    "output: just the puzzle text",
    "don't include the answer",
    "the answer is",
    "your task:",
    "important:",
    "step by step",
    "```",
)


def normalize_text(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def is_ascii_text(text):
    return all(ord(char) < 128 for char in text)


def clean_generated_text(text):
    text = text.replace("```", "").strip()
    text = re.sub(r"^\s*(puzzle|prompt)\s*:\s*", "", text, flags=re.IGNORECASE)
    for spec in CATEGORY_SPECS.values():
        inline_header = f"{spec['intro']} {spec['section_header']}"
        text = text.replace(inline_header, f"{spec['intro']}\n\n{spec['section_header']}")
    text = re.sub(r"\s+Now, ", "\n\nNow, ", text)
    return text.strip()


def load_category_prompt_index(train_csv_path):
    prompts_by_category = {name: [] for name in CATEGORY_SPECS}

    with open(train_csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            prompt = row["prompt"]
            for name, spec in CATEGORY_SPECS.items():
                if spec["match_phrase"] in prompt:
                    prompts_by_category[name].append(prompt)
                    break

    return prompts_by_category


def load_jsonl_records(path):
    records = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except FileNotFoundError:
        return []
    return records


def to_roman(number):
    values = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]
    remaining = number
    output = []
    for value, symbol in values:
        while remaining >= value:
            output.append(symbol)
            remaining -= value
    return "".join(output)


def _normalized_text_answer(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _first_float_token(value: str) -> float | None:
    """Best-effort float for comparing numeric Wonderland answers (may include units)."""
    text = (value or "").strip().replace(",", "")
    lowered = text.lower()
    for suffix in (" meters", " meter", " metres", " metre", " m"):
        if lowered.endswith(suffix):
            text = text[: -len(suffix)].strip()
            lowered = text.lower()
            break
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def answers_equivalent(expected: str, candidate: str, puzzle_type: str) -> bool:
    """
    True if a model's \\boxed{} extraction is equivalent to generator ground truth.

    Used when ``expected_answer`` is stored at generation time (deterministic puzzles).
    """
    if candidate is None:
        return False
    exp = (expected or "").strip()
    got = (candidate or "").strip()
    if not exp or not got:
        return False

    if puzzle_type in ("Gravitational Constant", "Unit Conversion"):
        exp_f = _first_float_token(exp)
        got_f = _first_float_token(got)
        if exp_f is None or got_f is None:
            return False
        return abs(exp_f - got_f) <= max(1e-6, abs(exp_f) * 1e-9)

    if puzzle_type == "Bit Manipulation":
        e = re.sub(r"\s+", "", exp)
        g = re.sub(r"\s+", "", got)
        if len(e) != 8 or len(g) != 8:
            return False
        if not set(e) <= {"0", "1"} or not set(g) <= {"0", "1"}:
            return False
        return e == g

    if puzzle_type == "Numeral Conversion":
        return exp.upper() == got.upper()

    if puzzle_type == "Text Encryption":
        return _normalized_text_answer(exp) == _normalized_text_answer(got)

    if puzzle_type == "Equation Transformation":
        return exp.strip() == got.strip()

    return exp == got


def _validate_text_substitution(example_pairs):
    cipher_to_plain = {}
    plain_to_cipher = {}

    for cipher_text, plain_text in example_pairs:
        if len(cipher_text) != len(plain_text):
            return "text example changes length"

        for cipher_char, plain_char in zip(cipher_text, plain_text):
            if cipher_char == " " or plain_char == " ":
                if cipher_char != plain_char:
                    return "text example misaligns spaces"
                continue

            if cipher_char in cipher_to_plain and cipher_to_plain[cipher_char] != plain_char:
                return "text examples do not use a consistent substitution"
            if plain_char in plain_to_cipher and plain_to_cipher[plain_char] != cipher_char:
                return "text examples are not one-to-one"

            cipher_to_plain[cipher_char] = plain_char
            plain_to_cipher[plain_char] = cipher_char

    return None


def _validate_gravity_consistency(example_pairs):
    g_values = []
    for time_s, distance_m in example_pairs:
        time_value = float(time_s)
        distance_value = float(distance_m)
        if time_value <= 0:
            return "gravity example uses non-positive time"
        g_values.append((2.0 * distance_value) / (time_value * time_value))

    if max(g_values) - min(g_values) > 0.02:
        return "gravity examples are inconsistent with a single hidden g"
    return None


def _validate_unit_consistency(example_pairs):
    ratios = []
    for measurement, converted in example_pairs:
        measurement_value = float(measurement)
        converted_value = float(converted)
        if measurement_value <= 0:
            return "unit example uses non-positive measurement"
        ratios.append(converted_value / measurement_value)

    if max(ratios) - min(ratios) > 0.02:
        return "unit examples are inconsistent with one conversion factor"
    return None


def _validate_numeral_examples(example_pairs):
    for number_text, numeral in example_pairs:
        number = int(number_text)
        if not 1 <= number <= 100:
            return "numeral example is outside the train-set number range"
        if to_roman(number) != numeral:
            return "numeral example is not valid Roman numeral conversion"
    return None


def validate_puzzle_text(
    text,
    category_name,
    reference_prompts=None,
    seen_prompts=None,
):
    issues = []
    spec = CATEGORY_SPECS[category_name]
    cleaned_text = clean_generated_text(text)

    if len(cleaned_text) < 140:
        issues.append("too short")
        return issues

    if not is_ascii_text(cleaned_text):
        issues.append("contains non-ASCII characters")

    lowered_text = cleaned_text.lower()
    for fragment in GENERIC_BANNED_FRAGMENTS + spec["banned_fragments"]:
        if fragment and fragment in lowered_text:
            issues.append(f"contains banned fragment: {fragment}")

    lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]
    if len(lines) < 4:
        issues.append("not enough non-empty lines")
        return issues

    if lines[0] != spec["intro"]:
        issues.append("wrong intro line")

    if lines[1] != spec["section_header"]:
        issues.append("wrong section header")

    example_lines = lines[2:-1]
    question_line = lines[-1]

    if not (spec["min_examples"] <= len(example_lines) <= spec["max_examples"]):
        issues.append(
            f"expected {spec['min_examples']}-{spec['max_examples']} examples, "
            f"found {len(example_lines)}"
        )

    parsed_examples = []
    for line in example_lines:
        match = spec["example_pattern"].fullmatch(line)
        if not match:
            issues.append(f"bad example line: {line}")
            continue
        parsed_examples.append(match.groups())

    question_match = spec["question_pattern"].fullmatch(question_line)
    if not question_match:
        issues.append("bad final question line")

    if category_name == "Text Encryption" and parsed_examples:
        substitution_issue = _validate_text_substitution(parsed_examples)
        if substitution_issue:
            issues.append(substitution_issue)

    if category_name == "Gravitational Constant" and parsed_examples:
        gravity_issue = _validate_gravity_consistency(parsed_examples)
        if gravity_issue:
            issues.append(gravity_issue)

    if category_name == "Unit Conversion" and parsed_examples:
        unit_issue = _validate_unit_consistency(parsed_examples)
        if unit_issue:
            issues.append(unit_issue)

    if category_name == "Numeral Conversion" and parsed_examples:
        numeral_issue = _validate_numeral_examples(parsed_examples)
        if numeral_issue:
            issues.append(numeral_issue)

    if category_name == "Bit Manipulation" and parsed_examples and question_match:
        seen_inputs = {source for source, _ in parsed_examples}
        seen_outputs = {target for _, target in parsed_examples}
        if len(seen_inputs) != len(parsed_examples):
            issues.append("reuses a bit-manipulation input example")
        if len(seen_outputs) == 1:
            issues.append("all bit-manipulation outputs are identical")
        query_input = question_match.group(1)
        if query_input in seen_inputs:
            issues.append("question input already appears in examples")

    if category_name == "Equation Transformation" and parsed_examples and question_match:
        query_expression = question_match.group(1)
        example_inputs = {source for source, _ in parsed_examples}
        if query_expression in example_inputs:
            issues.append("question expression already appears in examples")

    max_train_example_overlap = spec.get("max_train_example_overlap", 3)
    same_query_overlap_limit = spec.get("same_query_overlap_limit", 2)

    if reference_prompts:
        normalized_references = {normalize_text(prompt) for prompt in reference_prompts}
        normalized_text = normalize_text(cleaned_text)
        if normalized_text in normalized_references:
            issues.append("exact duplicate of a train prompt")
        else:
            generated_example_set = set(example_lines)
            for prompt in reference_prompts:
                prompt_lines = [line.strip() for line in prompt.splitlines() if line.strip()]
                if len(prompt_lines) < 4:
                    continue

                train_example_lines = prompt_lines[2:-1]
                train_example_set = set(train_example_lines)
                train_question_line = prompt_lines[-1]
                overlap_count = len(generated_example_set & train_example_set)
                same_question = question_line == train_question_line

                if overlap_count >= max_train_example_overlap:
                    issues.append(
                        f"too much example overlap with train prompt ({overlap_count} lines)"
                    )
                    break

                if same_question and overlap_count >= same_query_overlap_limit:
                    issues.append(
                        "train prompt overlap is too high for the same final query"
                    )
                    break

    if seen_prompts:
        normalized_seen = {normalize_text(prompt) for prompt in seen_prompts}
        if normalize_text(cleaned_text) in normalized_seen:
            issues.append("duplicate of an already accepted generated prompt")

    return issues
