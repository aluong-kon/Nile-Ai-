"""Offline NLLB fine-tuning. Run on a GPU and keep a held-out test set."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

ds = load_dataset("dayomtechnologies/english-nuer-dinka-parallel-corpus")

BASE_MODEL = os.getenv("TRAIN_BASE_MODEL", "facebook/nllb-200-distilled-600M")
CORPUS_FILE = Path(os.getenv("CORPUS_FILE", "corpus/pairs.jsonl"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "artifacts/dinka-bor-nllb"))
HF_TOKEN = os.getenv("HF_TOKEN")
EN_LANG = os.getenv("EN_LANG", "eng_Latn")
DIK_LANG = os.getenv("DIK_LANG", "dik_Latn")


def load_pairs(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Training corpus not found: {path}")
    pairs, seen = [], set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
        en = str(row.get("english", row.get("src", ""))).strip()
        dik = str(row.get("dinka_bor", row.get("tgt", ""))).strip()
        if not en or not dik:
            continue
        key = (en.casefold(), dik.casefold())
        if key not in seen:
            seen.add(key)
            pairs.append({"english": en, "dinka_bor": dik})
    if len(pairs) < 20:
        raise ValueError(
            f"Only {len(pairs)} unique pairs found. This script requires at least 20 as a safety floor; "
            "for meaningful evaluation, collect substantially more verified pairs."
        )
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--push-to-hub", default=None, help="Optional Hugging Face model repo ID")
    args = parser.parse_args()

    if not 0.05 <= args.eval_ratio <= 0.3:
        raise ValueError("--eval-ratio must be between 0.05 and 0.3")

    pairs = load_pairs(CORPUS_FILE)
    rng = np.random.default_rng(42)
    order = rng.permutation(len(pairs))
    n_eval = max(1, int(len(pairs) * args.eval_ratio))
    eval_rows = [pairs[i] for i in order[:n_eval]]
    train_rows = [pairs[i] for i in order[n_eval:]]
    if len(train_rows) < 10:
        raise ValueError("Not enough training examples after creating the evaluation split.")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
    if not hasattr(tokenizer, "lang_code_to_id") or DIK_LANG not in tokenizer.lang_code_to_id:
        raise RuntimeError(
            f"{BASE_MODEL} tokenizer does not recognize {DIK_LANG!r}. "
            "A standard NLLB tokenizer may not support this Dinka tag. Confirm your Dinka-compatible tokenizer."
        )
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL, token=HF_TOKEN)
    model.config.forced_bos_token_id = tokenizer.lang_code_to_id[DIK_LANG]

    def encode(rows):
        ds = Dataset.from_list(rows)
        def tokenize(batch):
            tokenizer.src_lang = EN_LANG
            tokenizer.tgt_lang = DIK_LANG
            return tokenizer(
                batch["english"], text_target=batch["dinka_bor"],
                max_length=args.max_length, truncation=True,
            )
        return ds.map(tokenize, batched=True, remove_columns=ds.column_names)

    train_ds = encode(train_rows)
    eval_ds = encode(eval_rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
        push_to_hub=bool(args.push_to_hub),
        hub_model_id=args.push_to_hub,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        processing_class=tokenizer,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    (OUTPUT_DIR / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    if args.push_to_hub:
        trainer.push_to_hub()
        tokenizer.push_to_hub(args.push_to_hub)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "train_pairs": len(train_rows),
                      "eval_pairs": len(eval_rows), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
