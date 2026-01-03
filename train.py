import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config, GPT2LMHeadModel, get_linear_schedule_with_warmup
import wandb

CONFIG = {
    "data_path": "data",
    "batch_size": 64,
    "lr": 1e-4,
    "weight_decay": 0.0,
    "warmup_steps": 200,
    "epochs": 20,
    "eval_every": 200,
    "seed": 0,
    "n_layer": 4,
    "n_head": 4,
    "n_embd": 128,
    "n_positions": 32,
    "num_workers": 2,
    "device": "cuda",
    "grad_clip": 1.0,
    "val_limit": 2000,
    "wandb_project": "laura-ruis-test",
    "wandb_run_name": "lsd-addition-3",
    "wandb_mode": "online",
    "gen_batch_size": 64,
    "max_gen_tokens": 6,
}

# Tokenizing

class CharVocab:
    def __init__(self):
        tokens = [
            "<pad>",
            "<bos>",
            "<eos>",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "+",
            "=",
        ]
        self.tokens = tokens
        self.stoi = {t: i for i, t in enumerate(tokens)}
        self.itos = {i: t for i, t in enumerate(tokens)}
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]

    def encode_str(self, s: str) -> List[int]:
        return [self.stoi[ch] for ch in s]

    def decode(self, ids: Sequence[int], stop_at_eos: bool = True) -> str:
        out: List[str] = []
        for i in ids:
            tok = self.itos[int(i)]
            if tok == "<eos>" and stop_at_eos:
                break
            if tok in {"<pad>", "<bos>", "<eos>"}:
                continue
            out.append(tok)
        return "".join(out)


def encode_example(
    prompt: str, target: str, vocab: CharVocab, add_bos: bool = True, add_eos: bool = True
) -> Dict[str, torch.Tensor]:
    input_ids: List[int] = []
    labels: List[int] = []

    if add_bos:
        input_ids.append(vocab.bos_id)
        labels.append(-100)

    prompt_ids = vocab.encode_str(prompt)
    input_ids.extend(prompt_ids)
    labels.extend([-100] * len(prompt_ids))

    target_ids = vocab.encode_str(target)
    input_ids.extend(target_ids)
    labels.extend(target_ids)

    if add_eos:
        input_ids.append(vocab.eos_id)
        labels.append(vocab.eos_id)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def load_jsonl(path: Path, limit: int | None = None) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with path.open() as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            records.append(json.loads(line))
    return records

# Data

class AdditionDataset(Dataset):
    def __init__(
        self,
        path: Path,
        vocab: CharVocab,
        add_bos: bool = True,
        add_eos: bool = True,
        limit: int | None = None,
    ):
        self.records = load_jsonl(path, limit)
        self.vocab = vocab
        self.add_bos = add_bos
        self.add_eos = add_eos

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.records[idx]
        return encode_example(
            ex["prompt"], ex["target"], self.vocab, self.add_bos, self.add_eos
        )


def collate_batch(batch: List[Dict[str, torch.Tensor]], pad_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(len(ex["input_ids"]) for ex in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full_like(input_ids, -100)
    attention_mask = torch.zeros_like(input_ids)

    for i, ex in enumerate(batch):
        seq_len = len(ex["input_ids"])
        input_ids[i, :seq_len] = ex["input_ids"]
        labels[i, :seq_len] = ex["labels"]
        attention_mask[i, :seq_len] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }

# Training

def build_model(vocab: CharVocab, cfg: Dict) -> GPT2LMHeadModel:
    config = GPT2Config(
        vocab_size=len(vocab.tokens),
        n_positions=cfg["n_positions"],
        n_ctx=cfg["n_positions"],
        n_embd=cfg["n_embd"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        bos_token_id=vocab.bos_id,
        eos_token_id=vocab.eos_id,
        pad_token_id=vocab.pad_id,
    )
    return GPT2LMHeadModel(config)


def evaluate(
    model: GPT2LMHeadModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss

            token_count = (labels != -100).sum().item()
            total_loss += loss.item() * token_count
            total_tokens += token_count

    avg_loss = total_loss / total_tokens if total_tokens > 0 else math.nan
    return {"loss": avg_loss}


def chunked(iterable: List[Dict[str, object]], batch_size: int) -> Iterable[List[Dict[str, object]]]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def encode_prompt(prompt: str, vocab: CharVocab, add_bos: bool = True) -> torch.Tensor:
    ids: List[int] = []
    if add_bos:
        ids.append(vocab.bos_id)
    ids.extend(vocab.encode_str(prompt))
    return torch.tensor(ids, dtype=torch.long)


def train(cfg: Dict):
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    vocab = CharVocab()
    data_dir = Path(cfg["data_path"])
    train_ds = AdditionDataset(data_dir / "train.jsonl", vocab, limit=None)
    val_ds = AdditionDataset(data_dir / "val.jsonl", vocab, limit=cfg["val_limit"])

    collate = lambda batch: collate_batch(batch, vocab.pad_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=collate,
    )

    model = build_model(vocab, cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    total_steps = cfg["epochs"] * len(train_loader)
    warmup_steps = min(cfg["warmup_steps"], total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    wandb.init(
        project=cfg["wandb_project"],
        name=cfg["wandb_run_name"],
        mode=cfg["wandb_mode"],
        config=cfg,
    )

    step = 0
    for epoch in range(cfg["epochs"]):
        for batch in train_loader:
            model.train()
            optimizer.zero_grad(set_to_none=True)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["grad_clip"]
            ).item()

            optimizer.step()
            scheduler.step()

            step += 1
            lr = scheduler.get_last_lr()[0]

            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/grad_norm": grad_norm,
                    "train/lr": lr,
                    "step": step,
                },
                step=step,
            )

            if step % cfg["eval_every"] == 0:
                metrics = evaluate(
                    model,
                    val_loader,
                    device,
                )
                print(
                    f"step {step} | train_loss {loss.item():.4f} | "
                    f"val_loss {metrics['loss']:.4f}"
                )
                wandb.log(
                    {
                        "val/loss": metrics["loss"],
                        "step": step,
                    },
                    step=step,
                )

    # Final evaluation
    eval_loaders = {
        "val": val_loader,
        "test_holdout_a": DataLoader(
            AdditionDataset(data_dir / "test_holdout_a.jsonl", vocab, limit=None),
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg["num_workers"],
            collate_fn=collate,
        ),
        "test_4digit": DataLoader(
            AdditionDataset(data_dir / "test_4digit.jsonl", vocab, limit=None),
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg["num_workers"],
            collate_fn=collate,
        ),
        "test_leading_zero": DataLoader(
            AdditionDataset(data_dir / "test_leading_zero.jsonl", vocab, limit=None),
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg["num_workers"],
            collate_fn=collate,
        ),
    }

    model.eval()
    results = []
    for split_name, loader in eval_loaders.items():
        raw_records = load_jsonl(data_dir / f"{split_name}.jsonl", limit=None)
        # Align raw records order with loader batches
        for batch_records in chunked(raw_records, cfg["gen_batch_size"]):
            prompts = [encode_prompt(rec["prompt"], vocab) for rec in batch_records]
            max_len = max(len(p) for p in prompts)
            input_ids = torch.full(
                (len(prompts), max_len), vocab.pad_id, dtype=torch.long, device=device
            )
            attention_mask = torch.zeros_like(input_ids)
            prompt_lens: List[int] = []
            for i, p in enumerate(prompts):
                input_ids[i, : len(p)] = p.to(device)
                attention_mask[i, : len(p)] = 1
                prompt_lens.append(len(p))

            # Greedy!
            generations = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=cfg["max_gen_tokens"],
                do_sample=False,
                eos_token_id=vocab.eos_id,
                pad_token_id=vocab.pad_id,
            )

            for rec, gen, p_len in zip(batch_records, generations, prompt_lens):
                pred_tokens = gen[p_len:]
                pred_str = vocab.decode(pred_tokens.tolist())
                out = dict(rec)
                out["split"] = split_name
                out["pred"] = pred_str
                results.append(out)

    results_path = data_dir / "results_raw.jsonl"
    with results_path.open("w") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")
    print(f"Saved raw generations to {results_path}")

    wandb.finish()
    return model, vocab, data_dir


if __name__ == "__main__":
    model, vocab, data_dir = train(CONFIG)
