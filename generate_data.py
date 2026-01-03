import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

CFG = {
    "seed": 42,
    "k": 3,
    "a_range": (100, 999),
    "b_range": (100, 999),
    "holdout_a_band": (800, 899),
    "n_train": 200_000,
    "n_val": 20_000,
    "n_test_holdout_a": 5_000,
    "n_test_4digit": 5_000,
    "n_test_leading_zero": 5000,
    "out_dir": "data",
}

# Utils

def int_to_lsd_str(n: int) -> str:
    if n == 0:
        return "0"
    digits = []
    while n > 0:
        digits.append(str(n % 10))
        n //= 10
    return "".join(digits)


def int_to_lsd_str_fixed_k(n: int, k: int) -> str:
    digits = []
    x = n
    for _ in range(k):
        digits.append(str(x % 10))
        x //= 10
    return "".join(digits)


def lsd_str_to_int(s: str) -> int:
    total = 0
    place = 1
    for ch in s:
        total += int(ch) * place
        place *= 10
    return total


def carry_stats_add(a: int, b: int, k: int) -> Tuple[int, int]:
    carries: List[int] = []
    carry = 0
    aa, bb = a, b

    for _ in range(k):
        da = aa % 10
        db = bb % 10
        aa //= 10
        bb //= 10

        s = da + db + carry
        carry = 1 if s >= 10 else 0
        carries.append(carry)

    num_carries = sum(carries)

    max_chain = 0
    cur = 0
    for c in carries:
        if c == 1:
            cur += 1
            max_chain = max(max_chain, cur)
        else:
            cur = 0

    return num_carries, max_chain

def make_example(a: int, b: int, k: int) -> Dict[str, object]:
    s = a + b
    num_carries, max_chain = carry_stats_add(a, b, k)

    a_str = int_to_lsd_str_fixed_k(a, k)
    b_str = int_to_lsd_str_fixed_k(b, k)
    s_str = int_to_lsd_str(s)

    return {
        "prompt": f"{a_str}+{b_str}=",
        "target": s_str,
        "a": a,
        "b": b,
        "sum": s,
        "num_carries": num_carries,
        "max_carry_chain_len": max_chain,
    }

def write_jsonl(path: Path, examples: Iterable[Dict[str, object]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

def main():
    rng = random.Random(CFG["seed"])
    k = CFG["k"]
    a_range = CFG["a_range"]
    b_range = CFG["b_range"]
    holdout = CFG["holdout_a_band"]

    print("Generating train + val")
    # Build full pool excluding holdout band on a
    pool: List[Tuple[int, int]] = [
        (a, b)
        for a in range(a_range[0], a_range[1] + 1)
        if not (holdout[0] <= a <= holdout[1])
        for b in range(b_range[0], b_range[1] + 1)
    ]
    rng.shuffle(pool)

    train_pairs = pool[: CFG["n_train"]]
    val_pairs = pool[CFG["n_train"] : CFG["n_train"] + CFG["n_val"]]

    train = [make_example(a, b, k) for a, b in train_pairs]
    val = [make_example(a, b, k) for a, b in val_pairs]

    # Generalization splits (sampled without replacement)
    print("Generating test sets")

    print("Generating test_holdout_a")
    test_holdout_a = []
    seen: set[Tuple[int, int]] = set()
    while len(test_holdout_a) < CFG["n_test_holdout_a"]:
        a = rng.randint(*holdout)
        b = rng.randint(*b_range)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        test_holdout_a.append(make_example(a, b, k))

    print("Generating test_4digit")
    test_4digit = []
    seen.clear()
    while len(test_4digit) < CFG["n_test_4digit"]:
        a = rng.randint(1000, 9999)
        b = rng.randint(1000, 9999)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        test_4digit.append(make_example(a, b, 4))

    print("Generating test_leading_zero")
    test_leading_zero = []
    seen.clear()
    while len(test_leading_zero) < CFG["n_test_leading_zero"]:
        a = rng.randint(0, 99)
        b = rng.randint(0, 99)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        test_leading_zero.append(make_example(a, b, k))

    out_dir = Path(CFG["out_dir"])
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)
    write_jsonl(out_dir / "test_holdout_a.jsonl", test_holdout_a)
    write_jsonl(out_dir / "test_4digit.jsonl", test_4digit)
    write_jsonl(out_dir / "test_leading_zero.jsonl", test_leading_zero)

    print("Datasets written to:", out_dir.resolve())


if __name__ == "__main__":
    main()
