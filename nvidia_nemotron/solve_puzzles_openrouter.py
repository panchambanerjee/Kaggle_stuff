"""
Solve generated puzzle JSONL via OpenRouter (OpenAI-compatible chat completions).

Aligned with ``solve_puzzles_ollama.py --from-generated``:

- **Bit Manipulation**: ``expected_answer`` is ignored; save when **min-consensus**
  (default 2 of 3) agree on the same parsed answer.
- **Other types**: save only when the answer matches ``expected_answer``
  (``answers_equivalent``).

Setup:
    Add to ``.env`` next to this script (do not commit secrets):

        OPENROUTER_API_KEY=sk-or-v1-...

    pip install python-dotenv

Examples:
    python solve_puzzles_openrouter.py --from-generated --model google/gemini-2.5-flash \\
        -o generated_puzzles_solved_openrouter.jsonl

    python solve_puzzles_openrouter.py --test --model deepseek/deepseek-r1

    python convert_extra_cots.py --jsonl generated_puzzles_solved_openrouter.jsonl --in-place --backup
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from puzzle_quality import (
    answers_equivalent,
    build_solver_prompt,
    extract_model_answer,
)

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_INPUT = _SCRIPT_DIR / "generated_puzzles_unsolved.jsonl"
_DEFAULT_OUTPUT = _SCRIPT_DIR / "generated_puzzles_solved_openrouter.jsonl"
BIT_MANIPULATION = "Bit Manipulation"

TEMPERATURE = 0.7
ATTEMPTS_PER_PUZZLE = 3
MIN_CONSENSUS = 2
TIMEOUT = 120
MAX_TOKENS = 2048
DEBUG = True
VERBOSE_VERIFY = False
STRIP_REDACTED_THINKING_SUFFIX = True
API_SLEEP_S = 0.75

OPENROUTER_API_KEY = ""
OPENROUTER_REFERER = "https://localhost/kaggle-nemotron-puzzle-solver"
OPENROUTER_TITLE = "nemotron-puzzle-solver"


def _message_content_to_text(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text") or ""))
                elif "text" in block:
                    parts.append(str(block.get("text") or ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def merge_openrouter_choice_to_cot(choice: dict) -> str:
    """Build supervision string from OpenRouter ``choices[0]`` (thinking + content)."""
    msg = choice.get("message")
    if not isinstance(msg, dict):
        msg = {}

    reasoning_chunks: list[str] = []
    for key in ("reasoning", "reasoning_content", "thinking"):
        val = msg.get(key)
        if isinstance(val, str) and val.strip():
            reasoning_chunks.append(val.strip())
    thinking = "\n\n".join(reasoning_chunks)

    response = _message_content_to_text(msg.get("content")).strip()

    if STRIP_REDACTED_THINKING_SUFFIX and "</think>" in response:
        response = response.split("</think>")[-1].strip()

    if thinking and response:
        return f"{thinking}\n\n{response}"
    if thinking:
        return thinking
    return response


def openrouter_chat_completions(body: dict) -> dict | None:
    """POST JSON to OpenRouter; returns parsed JSON or None on failure."""
    if not OPENROUTER_API_KEY:
        if DEBUG:
            print("  missing OPENROUTER_API_KEY")
        return None

    data = json.dumps(body).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": OPENROUTER_REFERER,
        "X-Title": OPENROUTER_TITLE,
    }
    request = Request(
        OPENROUTER_CHAT_URL,
        data=data,
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=TIMEOUT) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw)
    except HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        if DEBUG:
            snippet = err_body[:400] if err_body else ""
            print(f"  HTTP {e.code}: {snippet!r}")
        return None
    except URLError as e:
        if DEBUG:
            print(f"  URLError: {e}")
        return None
    except Exception as e:
        if DEBUG:
            print(f"  Error: {e}")
        return None


def solve_puzzle_with_cot_openrouter(puzzle_text: str, model: str, dry_run: bool) -> str | None:
    """Single chat completion; returns merged CoT text or None."""
    prompt = build_solver_prompt(puzzle_text)
    if dry_run:
        if DEBUG:
            print(f"  [dry-run] prompt chars={len(prompt)} model={model!r}")
        return None

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    result = openrouter_chat_completions(body)
    time.sleep(API_SLEEP_S)

    if not result:
        return None
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices:
        if DEBUG:
            print("  OpenRouter response had no choices")
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    cot = merge_openrouter_choice_to_cot(first).strip()
    return cot or None


def solve_with_consensus_openrouter(
    puzzle_text: str,
    model: str,
    puzzle_type: str,
    dry_run: bool,
) -> tuple[str | None, str | None, float]:
    """
    Up to ATTEMPTS_PER_PUZZLE calls; require MIN_CONSENSUS agreeing on the same
    ``extract_model_answer`` (used for Bit Manipulation).
    """
    label = model[:24] + "…" if len(model) > 24 else model
    print(f"  [{label}] consensus (≤{ATTEMPTS_PER_PUZZLE})...", end=" ", flush=True)

    if dry_run:
        print("dry-run", flush=True)
        solve_puzzle_with_cot_openrouter(puzzle_text, model, dry_run=True)
        return None, None, 0.0

    attempts: list[dict] = []

    for i in range(ATTEMPTS_PER_PUZZLE):
        if DEBUG and i > 0:
            print(f"{i + 1}", end=" ", flush=True)

        cot = solve_puzzle_with_cot_openrouter(puzzle_text, model, dry_run=False)
        if not cot:
            continue

        guess = extract_model_answer(cot, puzzle_type)
        if guess:
            attempts.append({"answer": guess, "cot": cot, "length": len(cot)})

    if len(attempts) == 0:
        print("❌ all failed")
        return None, None, 0.0

    answers = [a["answer"] for a in attempts]
    answer_counts = Counter(answers)
    most_common_answer, count = answer_counts.most_common(1)[0]

    if count < MIN_CONSENSUS:
        print(f"❌ no consensus ({count}/{len(attempts)} agree; need {MIN_CONSENSUS})")
        if DEBUG:
            print(f"     Answers: {answers}")
        return None, None, 0.0

    consensus_attempts = [a for a in attempts if a["answer"] == most_common_answer]
    best_cot = max(consensus_attempts, key=lambda x: x["length"])["cot"]

    confidence = count / len(attempts)
    print(f"✅ ({count}/{len(attempts)} agree)")

    return most_common_answer, best_cot, confidence


def solve_with_expected_answer(
    puzzle_text: str,
    model: str,
    expected: str,
    puzzle_type: str,
    dry_run: bool,
) -> tuple[str | None, str | None, int]:
    """
    Up to ATTEMPTS_PER_PUZZLE generations; first answer equivalent to ``expected`` wins.
    Returns (canonical_answer, cot, attempts_used) or (None, None, 0).
    """
    label = model[:24] + "…" if len(model) > 24 else model
    print(f"  [{label}] verify (≤{ATTEMPTS_PER_PUZZLE})...", end=" ", flush=True)

    if dry_run:
        print("dry-run", flush=True)
        solve_puzzle_with_cot_openrouter(puzzle_text, model, dry_run=True)
        return None, None, 0

    last_guesses: list[str] = []

    for i in range(ATTEMPTS_PER_PUZZLE):
        if DEBUG and i > 0:
            print(f"{i + 1}", end=" ", flush=True)

        cot = solve_puzzle_with_cot_openrouter(puzzle_text, model, dry_run=False)

        if not cot:
            if VERBOSE_VERIFY:
                print(f"\n     try {i + 1}: empty model output")
            continue
        guess = extract_model_answer(cot, puzzle_type)
        if guess:
            last_guesses.append(guess)
        if guess and answers_equivalent(expected, guess, puzzle_type):
            print(f"✅ try {i + 1}")
            return (expected or "").strip(), cot, i + 1
        if VERBOSE_VERIFY and guess:
            print(f"\n     try {i + 1}: guess={guess!r} expected={expected!r}")
        elif VERBOSE_VERIFY and not guess:
            print(f"\n     try {i + 1}: no parseable answer ({len(cot)} chars)")

    print("❌ no matching answer for expected_answer")
    if DEBUG and last_guesses:
        print(f"     last guesses: {last_guesses[-3:]} | expected: {expected!r}")
    return None, None, 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Solve puzzle JSONL via OpenRouter. With --from-generated: Bit uses consensus; others verify expected_answer.",
    )
    p.add_argument(
        "--model",
        required=True,
        help="OpenRouter model slug (e.g. google/gemini-2.5-flash, deepseek/deepseek-r1)",
    )
    p.add_argument(
        "--cross-check-model",
        default=None,
        metavar="MODEL",
        help="Second model: Bit rows need same consensus answer; other types need both to match expected_answer.",
    )
    p.add_argument("--input", "-i", type=Path, default=_DEFAULT_INPUT, help="Input JSONL")
    p.add_argument("--output", "-o", type=Path, default=_DEFAULT_OUTPUT, help="Output JSONL (append)")
    p.add_argument("--temperature", type=float, default=TEMPERATURE)
    p.add_argument("--attempts", type=int, default=ATTEMPTS_PER_PUZZLE)
    p.add_argument(
        "--min-consensus",
        type=int,
        default=MIN_CONSENSUS,
        help=f"For Bit Manipulation with --from-generated: min agreeing attempts (default {MIN_CONSENSUS})",
    )
    p.add_argument("--timeout", type=int, default=TIMEOUT)
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    p.add_argument(
        "--sleep",
        type=float,
        default=API_SLEEP_S,
        metavar="SEC",
        help=f"Pause after each API call (default: {API_SLEEP_S})",
    )
    p.add_argument("--referer", default=OPENROUTER_REFERER, help="HTTP-Referer header for OpenRouter")
    p.add_argument("--app-title", default=OPENROUTER_TITLE, help="X-Title header for OpenRouter")
    p.add_argument("--no-debug", action="store_true", help="Less verbose stderr")
    p.add_argument(
        "--verbose-verify",
        action="store_true",
        help="On failures, print each wrong guess vs expected_answer",
    )
    p.add_argument(
        "--keep-redacted-thinking",
        action="store_true",
        help="Do not strip text before </think> inside message content.",
    )
    p.add_argument(
        "--from-generated",
        action="store_true",
        help="Synthetic pipeline: Bit uses min-consensus (ignores expected_answer); "
        "other types require expected_answer and verify against it.",
    )
    p.add_argument("--limit", type=int, default=None, metavar="N", help="Process only first N puzzles")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call API or write output; print prompt stats only.",
    )
    p.add_argument("--test", action="store_true", help="Smoke test with one toy puzzle (expected_answer verify)")
    args = p.parse_args(argv)

    if args.cross_check_model and args.cross_check_model == args.model:
        p.error("--cross-check-model must differ from --model")
    if args.attempts < 1:
        p.error("--attempts must be >= 1")
    if args.min_consensus < 1 or args.min_consensus > args.attempts:
        p.error("--min-consensus must be between 1 and --attempts")

    return args


def apply_runtime_config(args: argparse.Namespace) -> None:
    global TEMPERATURE, ATTEMPTS_PER_PUZZLE, MIN_CONSENSUS, TIMEOUT, MAX_TOKENS, DEBUG
    global VERBOSE_VERIFY, STRIP_REDACTED_THINKING_SUFFIX, API_SLEEP_S
    global OPENROUTER_REFERER, OPENROUTER_TITLE

    TEMPERATURE = args.temperature
    ATTEMPTS_PER_PUZZLE = args.attempts
    MIN_CONSENSUS = args.min_consensus
    TIMEOUT = args.timeout
    MAX_TOKENS = args.max_tokens
    DEBUG = not args.no_debug
    VERBOSE_VERIFY = bool(args.verbose_verify)
    STRIP_REDACTED_THINKING_SUFFIX = not args.keep_redacted_thinking
    API_SLEEP_S = max(0.0, args.sleep)
    OPENROUTER_REFERER = args.referer
    OPENROUTER_TITLE = args.app_title


def _load_api_key() -> str:
    load_dotenv(_SCRIPT_DIR / ".env")
    return (os.environ.get("OPENROUTER_API_KEY") or "").strip()


def main(args: argparse.Namespace) -> None:
    global OPENROUTER_API_KEY

    apply_runtime_config(args)
    OPENROUTER_API_KEY = _load_api_key()

    primary = args.model
    cross = args.cross_check_model
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not args.dry_run and not OPENROUTER_API_KEY:
        print("❌ OPENROUTER_API_KEY not set.", file=sys.stderr)
        print("   Add it to .env next to this script (see docstring).", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("PUZZLE SOLVER (OpenRouter → JSONL)")
    print("=" * 70)
    print(f"Primary model: {primary}")
    print("Mode: --from-generated")
    print(
        f"  Bit Manipulation: {MIN_CONSENSUS}/{ATTEMPTS_PER_PUZZLE} consensus (expected_answer ignored); "
        "other types: verify against expected_answer"
    )
    if cross:
        print(
            "Cross-check model: Bit — same consensus answer; other types — both match expected_answer"
        )
    print(f"Temperature: {TEMPERATURE} | max_tokens: {MAX_TOKENS} | timeout: {TIMEOUT}s")
    print(f"Sleep after each API call: {API_SLEEP_S}s")
    print(f"Strip </think> prefix in content: {STRIP_REDACTED_THINKING_SUFFIX}")
    print(f"Attempts per puzzle (per model): {ATTEMPTS_PER_PUZZLE}")
    print(f"Minimum consensus (Bit): {MIN_CONSENSUS}/{ATTEMPTS_PER_PUZZLE}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    if args.dry_run:
        print("Dry-run: no API calls, no writes.")
    print()

    print(f"Loading {input_path}...")
    try:
        text = input_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"❌ {input_path} not found!")
        print("   Run: python generate_puzzles.py first")
        return

    puzzles: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        puzzles.append(json.loads(line))

    if args.limit is not None:
        puzzles = puzzles[: max(0, args.limit)]

    print(f"✓ Loaded {len(puzzles)} puzzles to solve\n")

    with_expected = sum(
        1
        for p in puzzles
        if p.get("expected_answer") is not None and str(p.get("expected_answer", "")).strip() != ""
    )
    solvable = sum(
        1
        for p in puzzles
        if p.get("type") == BIT_MANIPULATION
        or (
            p.get("expected_answer") is not None
            and str(p.get("expected_answer", "")).strip() != ""
        )
    )
    if solvable == 0:
        print(
            "❌ No solvable rows (need Bit Manipulation rows and/or non-Bit rows with expected_answer).\n"
            "   Re-run: python generate_puzzles.py",
        )
        return
    if with_expected < len(puzzles):
        print(
            f"⚠️  {len(puzzles) - with_expected} rows lack expected_answer "
            f"(non-Bit rows will be skipped; Bit rows use consensus).\n"
        )
    else:
        print(
            f"All {len(puzzles)} rows carry expected_answer "
            f"(Bit: consensus {MIN_CONSENSUS}/{ATTEMPTS_PER_PUZZLE}; others: verify).\n"
        )

    total_solved = 0
    total_failed = 0
    total_skipped = 0

    for i, puzzle in enumerate(puzzles, 1):
        print(f"\n[{i}/{len(puzzles)}] {puzzle['id']}")
        print(f"  Type: {puzzle['type']}")
        print(f"  Puzzle: {puzzle['prompt'][:60]}...")

        use_verify = bool(
            puzzle.get("expected_answer") is not None
            and str(puzzle.get("expected_answer", "")).strip() != ""
        )
        category = puzzle.get("type", "")
        expected_raw = (puzzle.get("expected_answer") or "").strip() if use_verify else ""
        is_bit = category == BIT_MANIPULATION
        bit_consensus_mode = is_bit
        expected_verify_mode = bool(use_verify and not is_bit)

        if not is_bit and not use_verify:
            print("  ⏭️  skip (non-Bit row missing expected_answer)")
            total_skipped += 1
            continue

        answer = None
        cot = None
        confidence = 0.0
        tries = ATTEMPTS_PER_PUZZLE
        cot_b = None
        confidence_b = 0.0
        tries_b = 0

        if bit_consensus_mode:
            answer, cot, confidence = solve_with_consensus_openrouter(
                puzzle["prompt"], primary, category, dry_run=args.dry_run
            )
            if cross:
                if not answer and not args.dry_run:
                    total_failed += 1
                    continue
                answer_b, cot_b, confidence_b = solve_with_consensus_openrouter(
                    puzzle["prompt"], cross, category, dry_run=args.dry_run
                )
                if not answer_b and not args.dry_run:
                    print("  ❌ Cross-check model did not reach a consensus answer")
                    total_failed += 1
                    continue
                if answer and answer_b and not answers_equivalent(
                    answer or "", answer_b or "", category
                ):
                    print(
                        f"  ❌ Models disagree on consensus answer: "
                        f"primary={answer!r} vs cross={answer_b!r}"
                    )
                    if not args.dry_run:
                        total_failed += 1
                    continue
                if answer and answer_b:
                    print("  ✅ Both models agree on the same consensus answer")
        else:
            answer, cot, tries = solve_with_expected_answer(
                puzzle["prompt"], primary, expected_raw, category, dry_run=args.dry_run
            )
            confidence = tries / ATTEMPTS_PER_PUZZLE if tries else 0.0

            if cross:
                if not answer and not args.dry_run:
                    total_failed += 1
                    continue
                answer_b, cot_b, tries_b = solve_with_expected_answer(
                    puzzle["prompt"], cross, expected_raw, category, dry_run=args.dry_run
                )
                confidence_b = tries_b / ATTEMPTS_PER_PUZZLE if tries_b else 0.0
                if not answer_b and not args.dry_run:
                    print("  ❌ Cross-check model did not produce a matching answer")
                    total_failed += 1
                    continue
                if answer and answer_b:
                    print("  ✅ Both models matched expected_answer")

        if answer:
            solved_puzzle = {
                **puzzle,
                "answer": answer,
                "generated_cot": cot,
                "consensus_confidence": confidence,
                "solver_temperature": TEMPERATURE,
                "attempts": ATTEMPTS_PER_PUZZLE,
                "status": "solved",
            }
            if bit_consensus_mode:
                solved_puzzle["solver_mode"] = "bit_consensus"
            else:
                solved_puzzle["solver_mode"] = "expected_verify"
                solved_puzzle["verification_attempts_primary"] = tries
                if cross:
                    solved_puzzle["verification_attempts_cross_check"] = tries_b

            if cross:
                cross_label = "bit_consensus" if bit_consensus_mode else "expected_verify"
                solved_puzzle["solved_by"] = f"{primary} + {cross} ({cross_label})"
                solved_puzzle["primary_model"] = primary
                solved_puzzle["cross_check_model"] = cross
                solved_puzzle["generated_cot_cross_check"] = cot_b
                solved_puzzle["consensus_confidence_cross_check"] = confidence_b
            else:
                solved_puzzle["solved_by"] = primary

            if not args.dry_run:
                with open(output_path, "a", encoding="utf-8") as handle:
                    json.dump(solved_puzzle, handle)
                    handle.write("\n")

            if DEBUG:
                print(f"     Answer: {answer}")
                if expected_verify_mode:
                    print(f"     Tries (primary): {tries}")
                    if cross:
                        print(f"     Tries (cross-check): {tries_b}")
                else:
                    print(f"     Confidence (primary): {confidence * 100:.0f}%")
                    if cross:
                        print(f"     Confidence (cross-check): {confidence_b * 100:.0f}%")

            total_solved += 1
        else:
            if not args.dry_run:
                total_failed += 1

        if i % 10 == 0:
            success_rate = (total_solved / i) * 100
            print(f"\n  📊 Progress: {total_solved}/{i} solved ({success_rate:.1f}%)")

    print(f"\n{'=' * 70}")
    print("SOLVING COMPLETE!")
    print(f"{'=' * 70}")
    print(f"Total puzzles in batch: {len(puzzles)}")
    if args.dry_run:
        print("Dry-run: no rows written")
    else:
        print(f"Written (solved rows): {total_solved}")
        print(f"Failed (no consensus / verify match): {total_failed}")
    if total_skipped:
        print(f"Skipped (non-Bit without expected_answer): {total_skipped}")
    denom = len(puzzles) - total_skipped
    if denom > 0:
        print(f"Yield vs attempted rows: {100.0 * total_solved / denom:.1f}%")
    print(f"\nSaved to: {output_path}")
    print()
    print("Next steps:")
    print("  python convert_extra_cots.py --jsonl <this-output> --in-place --backup")
    print(f"{'=' * 70}")


def test_mode(args: argparse.Namespace) -> None:
    global OPENROUTER_API_KEY

    apply_runtime_config(args)
    OPENROUTER_API_KEY = _load_api_key()

    primary = args.model
    cross = args.cross_check_model

    if not args.dry_run and not OPENROUTER_API_KEY:
        print("❌ OPENROUTER_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("TEST MODE (OpenRouter, expected_answer verify)")
    print("=" * 70)
    print()

    test_puzzle = """In a secret mathematical system, a transformation rule is applied to numbers.

Example 1: 5 becomes 25
Example 2: 3 becomes 9
Example 3: 7 becomes 49

In Alice's Wonderland, what does 6 become?"""

    expected = "36"
    puzzle_type = ""

    print("\nPuzzle:")
    print(test_puzzle)
    print(f"\nExpected answer (verify): {expected!r}\n")

    if args.dry_run:
        print("(dry-run: no API calls)\n")
        solve_puzzle_with_cot_openrouter(test_puzzle, primary, dry_run=True)
        if cross:
            solve_puzzle_with_cot_openrouter(test_puzzle, cross, dry_run=True)
        print("\n✓ Dry-run complete.")
        return

    answer, cot, tries = solve_with_expected_answer(
        test_puzzle, primary, expected, puzzle_type, dry_run=False
    )

    if cross and answer:
        answer_b, cot_b, tries_b = solve_with_expected_answer(
            test_puzzle, cross, expected, puzzle_type, dry_run=False
        )
        if not answer_b:
            print("\n❌ Cross-check model did not match expected_answer")
            return
        print("\n✅ Both models matched expected_answer")

    if answer:
        print("\n✅ Verified successfully!")
        print(f"Answer: {answer}")
        print(f"Tries (primary): {tries}")
        if cross:
            print(f"Tries (cross-check): {tries_b}")
        print("\nReasoning preview (primary):")
        print("-" * 70)
        preview = (cot or "")[:300]
        print(preview + ("..." if len(cot or "") > 300 else ""))
        print("-" * 70)
        if cross and cot_b:
            print("\nReasoning preview (cross-check):")
            print("-" * 70)
            preview_b = cot_b[:300]
            print(preview_b + ("..." if len(cot_b) > 300 else ""))
            print("-" * 70)
    else:
        print("\n❌ Could not verify (see messages above)")


if __name__ == "__main__":
    raw = sys.argv[1:]
    if raw and raw[0] == "test":
        raw = ["--test"] + raw[1:]
    ns = parse_args(raw)

    if ns.test:
        test_mode(ns)
    else:
        if not ns.from_generated:
            print("❌ --from-generated is required for JSONL mode (verify-only).", file=sys.stderr)
            sys.exit(2)
        main(ns)
