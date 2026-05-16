"""
SCRIPT 2: Solve Generated Puzzles with CoT (Ollama)
====================================================

Primary workflow: JSONL from ``generate_puzzles.py`` includes ``expected_answer``.

Run with ``--from-generated``:

- **Bit Manipulation**: ignore ``expected_answer`` (rules can be ambiguous). Save
  when **min-consensus** (default 2 of 3 attempts) agree on the same parsed answer
  (``extract_model_answer``).
- **All other types**: save only when the model answer matches ``expected_answer``
  (``answers_equivalent``).

Without ``--from-generated``, rows without ``expected_answer`` use the multi-attempt
**consensus** path (no oracle).

Optional ``--cross-check-model``: with ``--from-generated``, Bit rows require both
models to reach the same consensus answer; other types require both to match
``expected_answer``.

``generated_cot`` uses Ollama ``/api/chat``. Qwen3.5 Small (e.g. ``qwen3.5:4b``) has
**thinking off by default** (Unsloth); do not pass ``--think`` unless you want slow
hidden reasoning. Use ``--think`` only when you explicitly want thinking mode.

SETUP:
    ollama pull qwen3.5:4b
    python solve_puzzles_ollama.py --from-generated -o generated_puzzles_solved.jsonl
    python solve_puzzles_ollama.py --from-generated --model deepseek-r1:8b --think -o out.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from puzzle_quality import (
    answers_equivalent,
    build_solver_prompt,
    extract_boxed_answer,
    extract_model_answer,
)


# ============================================================
# CONFIGURATION (defaults; overridden by CLI)
# ============================================================

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
_DEFAULT_MODEL = "qwen3.5:4b"
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_INPUT = _SCRIPT_DIR / "generated_puzzles_unsolved.jsonl"
_DEFAULT_OUTPUT = _SCRIPT_DIR / "generated_puzzles_solved.jsonl"
BIT_MANIPULATION = "Bit Manipulation"

# Solving settings (CLI can override)
TEMPERATURE = 0.7
ATTEMPTS_PER_PUZZLE = 3
MIN_CONSENSUS = 2
TIMEOUT = 600
NUM_PREDICT = 2048
DEBUG = True
VERBOSE_VERIFY = False
# Set in apply_runtime_config from CLI
OLLAMA_REQUEST_THINK = False
STRIP_REDACTED_THINKING_SUFFIX = True


# ============================================================
# OLLAMA HELPER
# ============================================================


def _ollama_model_tags() -> list[str]:
    request = Request("http://localhost:11434/api/tags")
    with urlopen(request, timeout=5) as response:
        if response.status != 200:
            return []
        payload = json.loads(response.read().decode("utf-8"))
    return [m.get("name", "") for m in payload.get("models", [])]


def _model_available(requested: str, available: list[str]) -> bool:
    if requested in available:
        return True
    # Ollama sometimes lists tags with variant suffixes
    for name in available:
        if name == requested or name.startswith(requested + ":"):
            return True
    return False


def test_ollama_connection(model_names: list[str]) -> bool:
    """Check Ollama is up and every listed model is available."""
    try:
        available = _ollama_model_tags()
        if not available:
            print("❌ Ollama not running or returned no models! Start: ollama serve")
            return False
        missing = [m for m in model_names if not _model_available(m, available)]
        if missing:
            for m in missing:
                print(f"❌ Model '{m}' not found!")
                print(f"   Pull: ollama pull {m}")
            return False
        print("✓ Connected to Ollama")
        for m in model_names:
            print(f"✓ Model '{m}' ready")
        return True
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False


# ============================================================
# SOLVING WITH COT
# ============================================================


def merge_ollama_response_to_cot(result: dict) -> str:
    """
    Build the supervision string stored as ``generated_cot``.

    Supports both ``/api/chat`` (message.content / message.thinking) and
    ``/api/generate`` (response / thinking) payloads.
    """
    if isinstance(result.get("message"), dict):
        msg = result["message"]
        thinking = (msg.get("thinking") or "").strip()
        response = msg.get("content") or ""
    else:
        thinking = (result.get("thinking") or "").strip()
        response = result.get("response") or ""

    if STRIP_REDACTED_THINKING_SUFFIX and "</think>" in response:
        response = response.split("</think>")[-1]
    response = response.strip()

    if thinking and response:
        return f"{thinking}\n\n{response}"
    if thinking:
        return thinking
    return response


def _model_prefers_think(model: str) -> bool:
    """
    Only Gemma 4 needs think:true by default in our experience.

    Qwen3.5 Small (0.8B–9B) has thinking **disabled** by default (Unsloth). Forcing
    think:true spends the whole token budget in hidden reasoning and causes timeouts.
    """
    return "gemma4" in model.lower()


def _ollama_sampling_options(model: str, use_think: bool) -> dict:
    """
    Per-model sampling. Qwen3.5 Unsloth guide:
    - non-thinking + reasoning: temp 1.0, top_p 0.95, top_k 20
    - thinking + coding: temp 0.6, top_p 0.95, top_k 20
    """
    name = model.lower()
    opts: dict = {"num_predict": NUM_PREDICT}
    if "qwen3.5" in name or name.startswith("qwen3"):
        if use_think:
            opts.update(temperature=0.6, top_p=0.95, top_k=20)
        else:
            # Unsloth non-thinking "general" profile (puzzles are hard but less rambling than temp 1.0)
            opts.update(temperature=0.7, top_p=0.8, top_k=20)
        return opts
    opts["temperature"] = TEMPERATURE
    opts["top_p"] = 0.9
    return opts


def _post_ollama(url: str, payload: dict) -> dict | None:
    """POST JSON to Ollama; on HTTP 400 with think, retry once without think."""
    tried_drop_think = False
    body = payload
    while True:
        try:
            request = Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=TIMEOUT) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as e:
            raw = e.read()
            err_body = raw.decode("utf-8", errors="replace") if raw else ""
            if e.code == 400 and body.get("think") is True and not tried_drop_think:
                tried_drop_think = True
                body = {k: v for k, v in body.items() if k != "think"}
                if DEBUG:
                    print("  HTTP 400 — retrying without think", end=" ", flush=True)
                continue
            if DEBUG:
                print(f"  HTTP {e.code}: {err_body[:300]!r}")
            return None
        except URLError as e:
            if DEBUG:
                print(f"  Error: {e}")
            return None
        except Exception as e:
            if DEBUG:
                print(f"  Error: {e}")
            return None


def _call_ollama_chat(payload: dict) -> dict | None:
    return _post_ollama(OLLAMA_CHAT_URL, payload)


def _call_ollama_generate(payload: dict) -> dict | None:
    return _post_ollama(OLLAMA_GENERATE_URL, payload)


def solve_puzzle_with_cot(puzzle_text: str, model: str) -> str | None:
    """Solve puzzle and generate CoT reasoning for one Ollama model."""

    prompt = build_solver_prompt(puzzle_text)
    use_think = OLLAMA_REQUEST_THINK or _model_prefers_think(model)

    def _one_call(num_predict: int, use_think_flag: bool) -> dict | None:
        options = _ollama_sampling_options(model, use_think_flag)
        options["num_predict"] = num_predict
        payload: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": options,
        }
        if use_think_flag:
            payload["think"] = True
        return _call_ollama_chat(payload)

    result = _one_call(NUM_PREDICT, use_think)
    if result:
        text = merge_ollama_response_to_cot(result).strip()
        if text:
            return text

    # Single fallback: shorter generation, still non-thinking for Qwen unless --think
    if DEBUG:
        print("  retry shorter...", end=" ", flush=True)
    result = _one_call(min(2048, NUM_PREDICT), use_think)
    if result:
        return merge_ollama_response_to_cot(result).strip() or None
    return None


# ============================================================
# CONSENSUS VALIDATION
# ============================================================


def solve_with_consensus(
    puzzle_text: str,
    puzzle_id: str,
    model: str,
    puzzle_type: str | None = None,
) -> tuple[str | None, str | None, float]:
    """
    Try to solve puzzle multiple times and use consensus for one model.
    If ``puzzle_type`` is set (e.g. Bit Manipulation), answers are taken from
    ``extract_model_answer``; otherwise from ``\\boxed{}`` only.
    Returns (answer, cot, confidence) or (None, None, 0)
    """
    label = model[:24] + "…" if len(model) > 24 else model
    print(f"  [{label}] {ATTEMPTS_PER_PUZZLE} attempts...", end=" ", flush=True)

    attempts = []

    for i in range(ATTEMPTS_PER_PUZZLE):
        if DEBUG and i > 0:
            print(f"{i + 1}", end=" ", flush=True)

        cot = solve_puzzle_with_cot(puzzle_text, model)
        if not cot:
            continue

        if puzzle_type:
            answer = extract_model_answer(cot, puzzle_type)
        else:
            answer = extract_boxed_answer(cot)
        if answer:
            attempts.append({"answer": answer, "cot": cot, "length": len(cot)})

    if len(attempts) == 0:
        print("❌ all failed")
        return None, None, 0.0

    answers = [a["answer"] for a in attempts]
    answer_counts = Counter(answers)
    most_common_answer, count = answer_counts.most_common(1)[0]

    if count < MIN_CONSENSUS:
        print(f"❌ no consensus ({count}/{len(attempts)} agree)")
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
) -> tuple[str | None, str | None, int]:
    """
    Up to ATTEMPTS_PER_PUZZLE generations; first \\boxed{} equivalent to ``expected`` wins.
    Returns (canonical_answer, cot, attempts_used) or (None, None, 0).
    """
    label = model[:24] + "…" if len(model) > 24 else model
    print(f"  [{label}] verify (≤{ATTEMPTS_PER_PUZZLE})...", end=" ", flush=True)

    last_guesses: list[str] = []

    for i in range(ATTEMPTS_PER_PUZZLE):
        if DEBUG and i > 0:
            print(f"{i + 1}", end=" ", flush=True)

        cot = solve_puzzle_with_cot(puzzle_text, model)
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


# ============================================================
# ARGUMENTS
# ============================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Solve puzzle JSONL with Ollama. Use --from-generated for generate_puzzles.py output.",
    )
    p.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=f"Primary Ollama model (default: {_DEFAULT_MODEL})",
    )
    p.add_argument(
        "--cross-check-model",
        default=None,
        metavar="MODEL",
        help="Second model: save only if primary and cross-check reach consensus and answers match exactly. "
        "Example tags: qwen3.5:4b, deepseek-r1:8b (use `ollama list` for yours).",
    )
    p.add_argument(
        "--input",
        "-i",
        type=Path,
        default=_DEFAULT_INPUT,
        help="Input JSONL",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Output JSONL (append mode)",
    )
    p.add_argument("--temperature", type=float, default=TEMPERATURE)
    p.add_argument("--attempts", type=int, default=ATTEMPTS_PER_PUZZLE)
    p.add_argument("--min-consensus", type=int, default=MIN_CONSENSUS)
    p.add_argument("--timeout", type=int, default=TIMEOUT)
    p.add_argument(
        "--num-predict",
        type=int,
        default=NUM_PREDICT,
        help="Ollama num_predict per call (default 4096; Unsloth suggests up to 32k but that is slow locally)",
    )
    p.add_argument("--no-debug", action="store_true", help="Less verbose stderr")
    p.add_argument(
        "--verbose-verify",
        action="store_true",
        help="On --from-generated failures, print each wrong \\boxed{} vs expected_answer",
    )
    p.add_argument(
        "--think",
        action="store_true",
        help="Enable Ollama thinking mode (think: true). Off by default. "
        "Qwen3.5-4B/9B Small models disable thinking by default — only use this if you "
        "want slow hidden reasoning traces.",
    )
    p.add_argument(
        "--keep-redacted-thinking",
        action="store_true",
        help="Do not strip text before </think> inside response (full response kept).",
    )
    p.add_argument(
        "--from-generated",
        action="store_true",
        help="Synthetic pipeline (generate_puzzles.py output). Bit Manipulation: "
        f"{MIN_CONSENSUS}/{ATTEMPTS_PER_PUZZLE} consensus (expected_answer ignored). "
        "Other types: skip rows without expected_answer; save when answer matches ground truth.",
    )
    p.add_argument(
        "--test",
        action="store_true",
        help="Smoke test with a toy puzzle (no JSONL I/O)",
    )
    args = p.parse_args(argv)

    if args.cross_check_model and args.cross_check_model == args.model:
        p.error("--cross-check-model must differ from --model")

    if args.attempts < 1:
        p.error("--attempts must be >= 1")
    if args.min_consensus < 1 or args.min_consensus > args.attempts:
        p.error("--min-consensus must be between 1 and --attempts")

    return args


def apply_runtime_config(args: argparse.Namespace) -> None:
    global TEMPERATURE, ATTEMPTS_PER_PUZZLE, MIN_CONSENSUS, TIMEOUT, DEBUG
    global OLLAMA_REQUEST_THINK, STRIP_REDACTED_THINKING_SUFFIX, NUM_PREDICT, VERBOSE_VERIFY
    TEMPERATURE = args.temperature
    ATTEMPTS_PER_PUZZLE = args.attempts
    MIN_CONSENSUS = args.min_consensus
    TIMEOUT = args.timeout
    NUM_PREDICT = args.num_predict
    DEBUG = not args.no_debug
    VERBOSE_VERIFY = bool(getattr(args, "verbose_verify", False))
    OLLAMA_REQUEST_THINK = bool(getattr(args, "think", False))
    STRIP_REDACTED_THINKING_SUFFIX = not getattr(args, "keep_redacted_thinking", False)


# ============================================================
# MAIN
# ============================================================


def main(args: argparse.Namespace) -> None:
    apply_runtime_config(args)
    primary = args.model
    cross = args.cross_check_model
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    print("=" * 70)
    print("PUZZLE SOLVER (Ollama → JSONL)")
    print("=" * 70)
    print(f"Primary model: {primary}")
    if args.from_generated:
        print(
            f"Mode: --from-generated (Bit: {MIN_CONSENSUS}/{ATTEMPTS_PER_PUZZLE} consensus, "
            "ignores expected_answer; other types: match expected_answer)"
        )
    if cross:
        print(
            f"Cross-check model: {cross} "
            f"(Bit: same consensus answer; other types with oracle: both match expected_answer)"
        )
    print(f"Temperature: {TEMPERATURE}")
    print(f"num_predict: {NUM_PREDICT} | timeout: {TIMEOUT}s")
    use_think_effective = OLLAMA_REQUEST_THINK or _model_prefers_think(primary)
    print(
        f"Ollama think: {use_think_effective} "
        f"(qwen3.5:4b defaults to non-thinking per Unsloth; pass --think to enable)"
    )
    print(f"Strip </think> prefix in response: {STRIP_REDACTED_THINKING_SUFFIX}")
    print(f"Attempts per puzzle (per model): {ATTEMPTS_PER_PUZZLE}")
    print(f"Minimum consensus: {MIN_CONSENSUS}/{ATTEMPTS_PER_PUZZLE} (Bit rows with --from-generated; legacy rows without expected_answer)")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print()

    models_needed = [primary] if not cross else [primary, cross]
    if not test_ollama_connection(models_needed):
        print("\n❌ Setup incomplete")
        return
    print()

    print(f"Loading {input_path}...")
    try:
        text = input_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"❌ {input_path} not found!")
        print("   Run: python generate_puzzles.py first")
        return

    puzzles = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        puzzles.append(json.loads(line))
    print(f"✓ Loaded {len(puzzles)} puzzles to solve\n")

    with_expected = sum(
        1
        for p in puzzles
        if p.get("expected_answer") is not None and str(p.get("expected_answer", "")).strip() != ""
    )
    if args.from_generated:
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
                "❌ --from-generated: no solvable rows (need Bit Manipulation rows and/or "
                "non-Bit rows with non-empty expected_answer).\n"
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
    elif with_expected:
        print(f"Verification mode: {with_expected}/{len(puzzles)} rows carry expected_answer (no consensus).\n")

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
        bit_consensus_mode = bool(args.from_generated and is_bit)
        expected_verify_mode = bool(use_verify and not bit_consensus_mode)

        if args.from_generated and not is_bit and not use_verify:
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
            answer, cot, confidence = solve_with_consensus(
                puzzle["prompt"], puzzle["id"], primary, category
            )
            if cross:
                if not answer:
                    total_failed += 1
                    continue
                answer_b, cot_b, confidence_b = solve_with_consensus(
                    puzzle["prompt"], puzzle["id"], cross, category
                )
                if not answer_b:
                    print("  ❌ Cross-check model did not reach a consensus answer")
                    total_failed += 1
                    continue
                if not answers_equivalent(answer or "", answer_b or "", category):
                    print(
                        f"  ❌ Models disagree on consensus answer: "
                        f"primary={answer!r} vs cross={answer_b!r}"
                    )
                    total_failed += 1
                    continue
                print("  ✅ Both models agree on the same consensus answer")
        elif expected_verify_mode:
            answer, cot, tries = solve_with_expected_answer(
                puzzle["prompt"], primary, expected_raw, category
            )
            confidence = tries / ATTEMPTS_PER_PUZZLE if tries else 0.0

            if cross:
                if not answer:
                    total_failed += 1
                    continue
                answer_b, cot_b, tries_b = solve_with_expected_answer(
                    puzzle["prompt"], cross, expected_raw, category
                )
                confidence_b = tries_b / ATTEMPTS_PER_PUZZLE if tries_b else 0.0
                if not answer_b:
                    print("  ❌ Cross-check model did not produce a matching boxed answer")
                    total_failed += 1
                    continue
                print("  ✅ Both models matched expected_answer")
        else:
            answer, cot, confidence = solve_with_consensus(
                puzzle["prompt"], puzzle["id"], primary, None
            )
            if cross:
                if not answer:
                    total_failed += 1
                    continue
                answer_b, cot_b, confidence_b = solve_with_consensus(
                    puzzle["prompt"], puzzle["id"], cross, None
                )
                if not answer_b:
                    print("  ❌ Cross-check model did not reach a consensus answer")
                    total_failed += 1
                    continue
                if answer != answer_b:
                    print(f"  ❌ Models disagree: primary={answer!r} vs cross={answer_b!r}")
                    total_failed += 1
                    continue
                print("  ✅ Both models agree on the same boxed answer")

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
            elif expected_verify_mode:
                solved_puzzle["solver_mode"] = "expected_verify"
                solved_puzzle["verification_attempts_primary"] = tries
                if cross:
                    solved_puzzle["verification_attempts_cross_check"] = tries_b
            else:
                solved_puzzle["solver_mode"] = "consensus"

            if cross:
                cross_label = (
                    "expected_verify"
                    if expected_verify_mode
                    else ("bit_consensus" if bit_consensus_mode else "agreement")
                )
                solved_puzzle["solved_by"] = f"{primary} + {cross} ({cross_label})"
                solved_puzzle["primary_model"] = primary
                solved_puzzle["cross_check_model"] = cross
                solved_puzzle["generated_cot_cross_check"] = cot_b
                solved_puzzle["consensus_confidence_cross_check"] = confidence_b
            else:
                solved_puzzle["solved_by"] = primary

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
            total_failed += 1

        if i % 10 == 0:
            success_rate = (total_solved / i) * 100
            print(f"\n  📊 Progress: {total_solved}/{i} solved ({success_rate:.1f}%)")

    print(f"\n{'=' * 70}")
    print("SOLVING COMPLETE!")
    print(f"{'=' * 70}")
    print(f"Total puzzles in file: {len(puzzles)}")
    print(f"Written (solved rows): {total_solved}")
    print(f"Failed (no valid match / consensus): {total_failed}")
    if total_skipped:
        print(f"Skipped (non-Bit without expected_answer): {total_skipped}")
    if len(puzzles) > 0 and not args.from_generated:
        success_rate = (total_solved / len(puzzles)) * 100
        print(f"Success rate (vs file size): {success_rate:.1f}%")
    elif len(puzzles) > 0 and args.from_generated:
        denom = len(puzzles) - total_skipped
        if denom > 0:
            print(f"Yield vs attempted rows: {100.0 * total_solved / denom:.1f}%")
    print(f"\nSaved to: {output_path}")
    print()
    print("Next steps:")
    print("  1. Review solved JSONL")
    print("  2. Merge into train_split_with_cot.csv:")
    print("       python convert_extra_cots.py --in-place --backup")
    print("     (or without --in-place to write train_split_with_cot_merged.csv)")
    print("  3. Use in model training")
    print(f"{'=' * 70}")


def test_mode(args: argparse.Namespace) -> None:
    apply_runtime_config(args)
    primary = args.model
    cross = args.cross_check_model

    print("=" * 70)
    print("TEST MODE")
    print("=" * 70)
    print()

    models_needed = [primary] if not cross else [primary, cross]
    if not test_ollama_connection(models_needed):
        return
    print()

    print("Testing with sample puzzle...")
    test_puzzle = """In a secret mathematical system, a transformation rule is applied to numbers.

Example 1: 5 becomes 25
Example 2: 3 becomes 9
Example 3: 7 becomes 49

In Alice's Wonderland, what does 6 become?"""

    print("\nPuzzle:")
    print(test_puzzle)
    print()

    answer, cot, confidence = solve_with_consensus(test_puzzle, "test", primary, None)

    if cross and answer:
        answer_b, cot_b, confidence_b = solve_with_consensus(test_puzzle, "test", cross, None)
        if not answer_b:
            print("\n❌ Cross-check model could not solve")
            return
        if answer != answer_b:
            print(f"\n❌ Models disagree: {answer!r} vs {answer_b!r}")
            return
        print("\n✅ Both models agree")

    if answer:
        print(f"\n✅ Solved successfully!")
        print(f"Answer: {answer}")
        print(f"Confidence (primary): {confidence * 100:.0f}%")
        if cross:
            print(f"Confidence (cross-check): {confidence_b * 100:.0f}%")
        print(f"\nReasoning preview (primary):")
        print("-" * 70)
        print(cot[:300] + "...")
        print("-" * 70)
        if cross:
            print(f"\nReasoning preview (cross-check):")
            print("-" * 70)
            print(cot_b[:300] + "...")
            print("-" * 70)
    else:
        print("\n❌ Could not solve")

    print()


if __name__ == "__main__":
    raw = sys.argv[1:]
    # Backward compatible: `python solve_puzzles_ollama.py test` → --test
    if raw and raw[0] == "test":
        raw = ["--test"] + raw[1:]
    ns = parse_args(raw)
    if ns.test:
        test_mode(ns)
    else:
        main(ns)
