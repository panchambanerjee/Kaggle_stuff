# NVIDIA Nemotron puzzle pipeline (synthetic CoT)

Python tools to **generate** Nemotron-style puzzles without an LLM, **validate** prompt shape against the competition train set, **solve** them with local **Ollama** or **OpenRouter**, and optionally **merge** verified rows into `train_split_with_cot.csv` for fine-tuning.

This folder is self-contained aside from `train.csv` (Kaggle train download) used for overlap checks during generation.

---

## End-to-end flow

```text
train.csv
    │
    ▼
generate_puzzles.py  ──►  generated_puzzles_unsolved.jsonl
    │
    ├──►  solve_puzzles_ollama.py --from-generated
    │         or
    └──►  solve_puzzles_openrouter.py --from-generated
    │
    ▼
generated_puzzles_solved*.jsonl
    │
    ▼
convert_extra_cots.py  ──►  train_split_with_cot.csv (append)
```

1. **Generate** deterministic puzzles + `expected_answer` (JSONL).  
2. **Solve** with a model; only high-confidence rows are written (see [Solver policy](#solver-policy---from-generated) below).  
3. **Merge** solved JSONL into your CoT training CSV (`convert_extra_cots.py`).

---

## Dependencies

| Need | Notes |
|------|--------|
| Python 3.10+ | stdlib + small deps |
| `train.csv` | Same directory as `generate_puzzles.py` for `puzzle_quality` overlap checks |
| **Ollama** | For `solve_puzzles_ollama.py`; install model tags you pass (e.g. `ollama pull qwen3.5:4b`) |
| **`python-dotenv`** | For `solve_puzzles_openrouter.py` (`.env` with `OPENROUTER_API_KEY`) |
| **`pandas`** | For `convert_extra_cots.py` only |

```bash
pip install python-dotenv pandas
```

---

## `generate_puzzles.py`

**Role:** Build synthetic puzzles **entirely in Python** (no LLM). Each category matches Nemotron prompt structure (Alice’s Wonderland framing, example blocks, final question line).

**Output:** `generated_puzzles_unsolved.jsonl` (append). Default **100 puzzles per category** (`NUM_PER_CATEGORY`).

**Categories:** Bit Manipulation, Gravitational Constant, Unit Conversion, Numeral Conversion, Text Encryption, Equation Transformation (see `CATEGORY_SPECS` in `puzzle_quality.py`).

**Each JSONL row includes:**

| Field | Meaning |
|--------|---------|
| `id` | Unique id (`gen_…`) |
| `prompt` | Full puzzle text (train-shaped) |
| `type` | Category name |
| `expected_answer` | Generator ground truth (string) |
| `generated_by` | e.g. `python-deterministic-bits` |
| `temperature` | Metadata echo from spec |
| `status` | `"unsolved"` until a solver persists a solved copy |

**Run:**

```bash
cd nvidia_nemotron
python generate_puzzles.py
```

**Smoke test (one puzzle per category, debug prints):**

```bash
python generate_puzzles.py test
```

**Requirements:** Validates every candidate with `validate_puzzle_text()` so prompts align with `train.csv` exemplars and internal consistency rules before append.

---

## `puzzle_quality.py`

**Role:** Shared **schemas, validation, and answer handling** for generation and solvers.

**Main pieces:**

| API | Purpose |
|-----|---------|
| `CATEGORY_SPECS` | Per-category intro text, regexes for question/examples, limits, banned fragments |
| `validate_puzzle_text()` | Structural + semantic checks vs train references |
| `build_solver_prompt()` | User message suffix (boxed instruction, step-by-step) — **must stay in sync** with both solvers |
| `extract_boxed_answer()` | Last `\boxed{...}` |
| `extract_model_answer()` | Prefer boxed; fallbacks (8-bit binary, Roman numerals, floats) by `type` |
| `answers_equivalent(expected, candidate, puzzle_type)` | Oracle match for non-Bit categories (float tolerance, bit equality, etc.) |
| `load_category_prompt_index` / `load_jsonl_records` | IO helpers for generation |

Solvers import this module; do not duplicate prompt or equivalence logic in model-facing code.

---

## `solve_puzzles_ollama.py`

**Role:** Call **local Ollama** (`/api/chat`) to produce chain-of-thought text, then **filter** rows before append-only JSONL output.

**Defaults:** `qwen3.5:4b`, input `generated_puzzles_unsolved.jsonl`, output `generated_puzzles_solved.jsonl`, temperature `0.7`, `num_predict` 2048, timeout 600s.

**Run (synthetic pipeline):**

```bash
ollama serve   # in another terminal
python solve_puzzles_ollama.py --from-generated -o generated_puzzles_solved.jsonl
```

**Useful flags:**

| Flag | Description |
|------|--------------|
| `--model` | Ollama model tag |
| `--from-generated` | Enable [solver policy](#solver-policy---from-generated) for `generate_puzzles.py` JSONL |
| `--cross-check-model` | Second model; stricter acceptance (see policy) |
| `--attempts`, `--min-consensus` | Attempt budget and Bit consensus threshold (defaults 3 and 2) |
| `--num-predict`, `--timeout` | Generation cap and HTTP timeout |
| `--think` | Ollama thinking mode (often slower; Qwen3.5 small is usually left off) |
| `--verbose-verify` | More detail when verification fails |
| `--test` | Toy puzzle + consensus smoke test (no JSONL) |

**Legacy mode:** Without `--from-generated`, rows **without** `expected_answer` use multi-attempt boxed consensus; rows **with** `expected_answer` use oracle verification.

**Connection:** On startup, checks `http://localhost:11434` and that requested model tags exist.

---

## `solve_puzzles_openrouter.py`

**Role:** Same **filtering policy** as Ollama for `--from-generated`, but calls **OpenRouter** OpenAI-compatible `POST /api/v1/chat/completions`.

**Requirements:**

- `.env` in this directory with `OPENROUTER_API_KEY=...` (never commit keys).
- `pip install python-dotenv`

**Run:**

```bash
python solve_puzzles_openrouter.py --from-generated --model google/gemini-2.5-flash \
  -o generated_puzzles_solved_openrouter.jsonl
```

**Flags of note:** `--model` is **required**; `--limit N` for partial runs; `--dry-run` (no API, no write); `--sleep` between calls to reduce 429s; `--min-consensus` for Bit; `--test` for a single oracle-style arithmetic check.

**Headers:** Sends `HTTP-Referer` and `X-Title` (overridable via `--referer` / `--app-title`) as recommended by OpenRouter.

---

## Solver policy (`--from-generated`)

When solving JSONL produced by `generate_puzzles.py`:

| `type` | Saved when |
|--------|------------|
| **Bit Manipulation** | **Consensus:** at least `--min-consensus` of `--attempts` runs agree on the same `extract_model_answer` result. **`expected_answer` is not used** (multiple bit rules can fit the same examples). |
| **All other types** | **Oracle:** `extract_model_answer` matches `expected_answer` via `answers_equivalent` within the attempt budget. |

**Cross-check (`--cross-check-model`):**

- **Bit:** both models must reach the **same** consensus answer.  
- **Non-Bit:** both must match `expected_answer`.

**Skips:** Non–Bit rows with empty `expected_answer` are skipped. Bit rows are always attempted.

**Output row metadata** (extra columns ignored by `convert_extra_cots.py`): e.g. `solver_mode` (`expected_verify` | `bit_consensus` | `consensus`), `solved_by`, `verification_attempts_*`, `consensus_confidence`, etc.

---

## Solved JSONL → training CSV

`convert_extra_cots.py` appends rows that contain at least: `id`, `prompt`, `answer`, `type`, `generated_cot`.

```bash
python convert_extra_cots.py --jsonl generated_puzzles_solved.jsonl --in-place --backup
```

Use `--dry-run` to print counts only.

---

## Other scripts (brief)

| File | Role |
|------|------|
| `filter_generated_puzzles.py` | Optional extra filtering after generation |
| `generate_missing_cots.py` | Legacy Gemini path for real `train.csv` equations |
| `training_with_unsloth_to_achieve_0_85_lb.py` | Training entrypoint (separate from solver pipeline) |

---

## Troubleshooting

| Symptom | Check |
|---------|--------|
| Ollama “model not found” | `ollama list` / `ollama pull <tag>` |
| Empty or truncated CoT | Raise `--num-predict` or `--max-tokens`; avoid `--think` on small Qwen unless needed |
| OpenRouter 401 / 429 | API key, `--sleep`, billing/limits |
| No rows merged | IDs may already exist in train CSV; `convert_extra_cots` dedupes by `id` |
| Bit never saves | Lower `--min-consensus` or raise `--attempts`; model may disagree across samples |

---

## File map

| File | Responsibility |
|------|----------------|
| `generate_puzzles.py` | Deterministic puzzle + answer generation → JSONL |
| `puzzle_quality.py` | Validation, prompts, answer extract/compare |
| `solve_puzzles_ollama.py` | Ollama chat solver + gating |
| `solve_puzzles_openrouter.py` | OpenRouter chat solver + gating |
| `convert_extra_cots.py` | JSONL → append `train_split_with_cot.csv` |
| `train.csv` | Reference prompts for quality gates |
| `train_split_with_cot.csv` | CoT training data (input to merge step) |
