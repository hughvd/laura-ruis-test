# Plan: 3-digit Addition with a Small Transformer (LSD-first)

## Current design choices

### Task
- Add two **3-digit** base-10 integers:
  - `a, b ∈ [100, 999]`
- Model input is an expression and the model outputs the sum.

### Representation / formatting
- **Least-significant-digit (LSD) first** representation for all numbers.
  - Example: `100 → "001"` (ones, tens, hundreds)
  - Example: `579 → "975"`
- **No padding beyond natural digit decomposition**:
  - Operands are always exactly 3 digits in LSD-first because `a,b` are 3-digit.
  - The sum can be 3 or 4 digits (max `1998`), so its LSD-first string is length 3 or 4 accordingly:
    - `200 → "002"` (length 3)
    - `1000 → "0001"` (length 4)

### Input / output strings
- Prompt (input): `"{rev(a)}+{rev(b)}="`
  - Example: `112 + 334 = 446` becomes `211+433=`
- Target (output): `rev(a+b)`
  - Example: `446 → "644"`
- The model should only be trained/evaluated on predicting the **answer tokens** (target), not the prompt.

### Sampling & splits (algorithmic generalization focus)
- **Train/val**: uniform sampling over pairs `(a,b)` with `a ∈ [100,999] \ [800,899]` (exclude held-out `a` band).
- **Test-ID**: same distribution as train/val (also excludes `a ∈ [800,899]`).
- **Test-holdout-a**: `a ∈ [800,899]`, `b ∈ [100,999]` (unseen `a` band), uniform.

*(Optional later)*: add extra evaluation sets for:
- 4-digit operands (length generalization)
- leading-zero variants / shorter numbers rendered as 3-digit LSD-first (e.g., `12 → "210"` vs canonical `012 → "210"` depending on chosen convention)

### Logging & diagnostics (per example)
We will log:
- `a`, `b`, `sum`
- `num_carries`
- `max_carry_chain_len`

Definitions (for 3-digit addition):
- We compute carry-out at each digit position (ones, tens, hundreds) plus the **final carry** into the thousands place.
- `num_carries` = count of carry-outs that are 1 across these positions (including final carry).
- `max_carry_chain_len` = longest run of consecutive carry-outs equal to 1.

### Metrics (evaluation)
For each eval slice:
- **Strict string match**: predicted string equals target string exactly.
- **Numeric match**: parse prediction as an integer (in base-10) and compare to true sum.
  - Parsing must interpret LSD-first digits correctly.

### Results artifact
- `results.json` should include metrics broken out by:
  - `test_id`
  - `test_holdout_a`
  - (and any later slices, e.g., by carry-count buckets)

## Code organization
- `generate_data.py`: standalone dataset generator producing JSONL files with fields:
  - `prompt`, `target`, `a`, `b`, `sum`, `num_carries`, `max_carry_chain_len`
- Outputs:
  - `data/train.jsonl`
  - `data/val.jsonl`
  - `data/test_id.jsonl`
  - `data/test_holdout_a.jsonl`

## Open knobs (not decided yet)
- Dataset sizes for each split (train/val/test)
- Whether to stratify training sampling by carry patterns vs just log and analyze them
- Tokenization strategy (character/digit-level vs other)
- Model size / training hyperparameters
