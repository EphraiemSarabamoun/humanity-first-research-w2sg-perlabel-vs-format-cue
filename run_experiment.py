import argparse
import gc
import json
import os
import random
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
    set_seed,
)


PDIR = Path(__file__).resolve().parent
RDIR = PDIR / "results" / "real"
POS = "positive"
NEG = "negative"
LABEL_TEXT = {1: POS, 0: NEG}
CONDITIONS = ["gold-ceiling", "real-weak", "shuffled-weak", "random-label", "constant-majority"]


class SentimentSFTDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.features = []
        eos = tokenizer.eos_token or ""
        for row in rows:
            prompt = make_prompt(row["sentence"], fewshot=False)
            completion = " " + LABEL_TEXT[int(row["label"])] + eos
            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            full_ids = tokenizer(prompt + completion, add_special_tokens=False).input_ids
            if len(full_ids) > max_length:
                full_ids = full_ids[-max_length:]
                prompt_len = min(len(prompt_ids), max_length - 1)
            else:
                prompt_len = len(prompt_ids)
            labels = list(full_ids)
            for i in range(min(prompt_len, len(labels))):
                labels[i] = -100
            self.features.append({"input_ids": full_ids, "labels": labels})

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


def collate_train(batch, tokenizer):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = []
    labels = []
    attention = []
    pad = tokenizer.pad_token_id
    for item in batch:
        n_pad = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad] * n_pad)
        labels.append(item["labels"] + [-100] * n_pad)
        attention.append([1] * len(item["input_ids"]) + [0] * n_pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
    }


def make_prompt(sentence, fewshot=False, shots=None):
    head = "Classify the sentiment of the movie review sentence. Answer with positive or negative.\n"
    if fewshot and shots:
        parts = [head]
        for ex in shots:
            parts.append(f"Sentence: {ex['sentence']}\nSentiment: {LABEL_TEXT[int(ex['label'])]}\n\n")
        parts.append(f"Sentence: {sentence}\nSentiment:")
        return "".join(parts)
    return head + f"Sentence: {sentence}\nSentiment:"


def parse_label(text):
    low = text.lower()
    hits = []
    for lab, val in [(POS, 1), (NEG, 0)]:
        idx = low.find(lab)
        if idx >= 0:
            hits.append((idx, val))
    if hits:
        return sorted(hits)[0][1]
    return None


def choose_label_from_logits(model, tokenizer, prompts, device):
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**enc)
    logits = out.logits[:, -1, :]
    pos_ids = tokenizer(" positive", add_special_tokens=False).input_ids
    neg_ids = tokenizer(" negative", add_special_tokens=False).input_ids
    pos_id = pos_ids[0]
    neg_id = neg_ids[0]
    return (logits[:, pos_id] >= logits[:, neg_id]).long().cpu().tolist()


def evaluate_model(model, tokenizer, rows, batch_size, max_length, fewshot_rows=None):
    model.eval()
    tokenizer.padding_side = "left"
    device = next(model.parameters()).device
    correct = 0
    total = 0
    parse_failures = 0
    preds = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = [
            make_prompt(r["sentence"], fewshot=bool(fewshot_rows), shots=fewshot_rows)
            for r in batch
        ]
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                do_sample=False,
                max_new_tokens=3,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = gen[:, enc["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        labels = []
        missing_indices = []
        for i, text in enumerate(decoded):
            lab = parse_label(text)
            if lab is None:
                parse_failures += 1
                missing_indices.append(i)
                labels.append(None)
            else:
                labels.append(lab)
        if missing_indices:
            fallback_prompts = [prompts[i] for i in missing_indices]
            fallback = choose_label_from_logits(model, tokenizer, fallback_prompts, device)
            for idx, lab in zip(missing_indices, fallback):
                labels[idx] = lab
        for row, pred in zip(batch, labels):
            preds.append(int(pred))
            correct += int(int(pred) == int(row["label"]))
            total += 1
    return {
        "accuracy": correct / total if total else 0.0,
        "predictions": preds,
        "parse_failures": parse_failures,
    }


def cleanup_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_tokenizer(model_name):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_model(model_name, use_4bit=True):
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": {"": 0},
        "trust_remote_code": True,
    }
    if use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def add_lora(model):
    if getattr(model, "is_loaded_in_4bit", False):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


def train_lora(model, tokenizer, train_rows, seed, max_length, epochs, batch_size, grad_accum, lr, max_steps=None):
    set_seed(seed)
    model.train()
    tokenizer.padding_side = "right"
    ds = SentimentSFTDataset(train_rows, tokenizer, max_length=max_length)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=lambda b: collate_train(b, tokenizer),
    )
    steps_per_epoch = (len(loader) + grad_accum - 1) // grad_accum
    total_steps = max(1, steps_per_epoch * epochs)
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    sched = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=max(1, int(total_steps * 0.03)),
        num_training_steps=total_steps,
    )
    device = next(model.parameters()).device
    global_step = 0
    last_loss = None
    optim.zero_grad(set_to_none=True)
    for _ in range(epochs):
        for batch_idx, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / grad_accum
            loss.backward()
            last_loss = float(out.loss.detach().cpu())
            if (batch_idx + 1) % grad_accum == 0:
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1
                if global_step >= total_steps:
                    return {"steps": global_step, "last_loss": last_loss}
        if global_step >= total_steps:
            break
    if global_step == 0:
        optim.step()
        sched.step()
        optim.zero_grad(set_to_none=True)
        global_step = 1
    return {"steps": global_step, "last_loss": last_loss}


def rows_from_split(split):
    return [{"idx": int(r["idx"]), "sentence": r["sentence"], "label": int(r["label"])} for r in split]


def split_data(seed, smoke):
    ds = load_dataset("nyu-mll/glue", "sst2")
    train = rows_from_split(ds["train"])
    val = rows_from_split(ds["validation"])
    rng = random.Random(seed)
    order = list(range(len(train)))
    rng.shuffle(order)
    weak_n = 300 if smoke else 3000
    k = 200 if smoke else 2000
    eval_n = 200 if smoke else len(val)
    weak_train = [train[i] for i in order[:weak_n]]
    student_gold = [train[i] for i in order[weak_n : weak_n + k]]
    eval_rows = val[:eval_n]
    return weak_train, student_gold, eval_rows


def label_accuracy(labels, gold_rows):
    return sum(int(a) == int(r["label"]) for a, r in zip(labels, gold_rows)) / len(gold_rows)


def build_condition_rows(student_gold, weak_labels, seed):
    gold_labels = [int(r["label"]) for r in student_gold]
    rng = random.Random(seed)
    labels = {}
    labels["gold-ceiling"] = gold_labels
    shuffled = list(weak_labels)
    rng.shuffle(shuffled)
    labels["shuffled-weak"] = shuffled
    labels["real-weak"] = list(weak_labels)
    labels["random-label"] = [rng.randrange(2) for _ in student_gold]
    majority = 1 if sum(gold_labels) >= len(gold_labels) / 2 else 0
    labels["constant-majority"] = [majority for _ in student_gold]
    out = {}
    for cond, labs in labels.items():
        out[cond] = [
            {"idx": row["idx"], "sentence": row["sentence"], "label": int(lab), "gold_label": int(row["label"])}
            for row, lab in zip(student_gold, labs)
        ]
    return out


def write_jsonl_row(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def run(args):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    RDIR.mkdir(parents=True, exist_ok=True)
    raw_path = RDIR / "raw_results.jsonl"
    raw_path.write_text("", encoding="utf-8")
    usage = {
        "run_id": f"w2sg-{int(time.time())}",
        "mode": "smoke" if args.smoke else "full",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "recorded_gpu_hours": 0.0,
        "per_experiment": [],
        "model_notes": [],
    }

    set_seed(args.seed)
    weak_train, student_gold, eval_rows = split_data(args.seed, args.smoke)
    seeds = [0] if args.smoke else list(range(5))
    conditions = ["gold-ceiling", "real-weak", "shuffled-weak"] if args.smoke else CONDITIONS

    student_model_name = args.student_model
    weak_model_name = args.weak_model
    student_tok = load_tokenizer(student_model_name)
    weak_tok = load_tokenizer(weak_model_name)

    t0 = time.monotonic()
    weak_model = add_lora(load_model(weak_model_name, use_4bit=True))
    weak_train_stats = train_lora(
        weak_model,
        weak_tok,
        weak_train,
        seed=args.seed,
        max_length=args.max_seq_len,
        epochs=1,
        batch_size=args.weak_batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        max_steps=args.smoke_steps if args.smoke else None,
    )
    weak_eval = evaluate_model(weak_model, weak_tok, eval_rows, args.eval_batch_size, args.max_seq_len)
    weak_student_pred_rows = evaluate_model(
        weak_model, weak_tok, student_gold, args.eval_batch_size, args.max_seq_len
    )
    weak_labels = weak_student_pred_rows["predictions"]
    weak_labeler_val_accuracy = weak_eval["accuracy"]
    usage["per_experiment"].append(
        {
            "name": "weak-labeler",
            "gpu_hours": (time.monotonic() - t0) / 3600.0,
            "train_steps": weak_train_stats["steps"],
            "last_loss": weak_train_stats["last_loss"],
            "val_accuracy": weak_labeler_val_accuracy,
            "parse_failures": weak_eval["parse_failures"],
        }
    )
    cleanup_model(weak_model)

    fewshot_rows = []
    by_label = {0: [], 1: []}
    for row in weak_train:
        by_label[int(row["label"])].append(row)
    for i in range(4):
        for lab in [0, 1]:
            if i < len(by_label[lab]):
                fewshot_rows.append(by_label[lab][i])

    t0 = time.monotonic()
    base_model = load_model(student_model_name, use_4bit=True)
    base_eval = evaluate_model(base_model, student_tok, eval_rows, args.eval_batch_size, args.max_seq_len, fewshot_rows)
    base_row = {
        "condition": "base-fewshot",
        "seed": -1,
        "student_val_accuracy": base_eval["accuracy"],
        "train_label_gold_accuracy": None,
        "weak_labeler_val_accuracy": weak_labeler_val_accuracy,
        "student_model": student_model_name,
        "weak_model": weak_model_name,
        "n_train": 0,
        "n_eval": len(eval_rows),
        "parse_failures": base_eval["parse_failures"],
        "mode": "smoke" if args.smoke else "full",
    }
    write_jsonl_row(raw_path, base_row)
    usage["per_experiment"].append(
        {"name": "base-fewshot", "gpu_hours": (time.monotonic() - t0) / 3600.0, "val_accuracy": base_eval["accuracy"]}
    )
    cleanup_model(base_model)

    for seed in seeds:
        label_sets = build_condition_rows(student_gold, weak_labels, seed)
        for cond in conditions:
            train_rows = label_sets[cond]
            t0 = time.monotonic()
            set_seed(seed)
            model = add_lora(load_model(student_model_name, use_4bit=True))
            train_stats = train_lora(
                model,
                student_tok,
                train_rows,
                seed=seed,
                max_length=args.max_seq_len,
                epochs=1 if args.smoke else args.epochs,
                batch_size=args.student_batch_size,
                grad_accum=args.grad_accum,
                lr=args.lr,
                max_steps=args.smoke_steps if args.smoke else None,
            )
            eval_stats = evaluate_model(model, student_tok, eval_rows, args.eval_batch_size, args.max_seq_len)
            train_acc = label_accuracy([r["label"] for r in train_rows], student_gold)
            row = {
                "condition": cond,
                "seed": seed,
                "student_val_accuracy": eval_stats["accuracy"],
                "train_label_gold_accuracy": train_acc,
                "weak_labeler_val_accuracy": weak_labeler_val_accuracy,
                "student_model": student_model_name,
                "weak_model": weak_model_name,
                "n_train": len(train_rows),
                "n_eval": len(eval_rows),
                "train_steps": train_stats["steps"],
                "last_loss": train_stats["last_loss"],
                "parse_failures": eval_stats["parse_failures"],
                "mode": "smoke" if args.smoke else "full",
            }
            write_jsonl_row(raw_path, row)
            elapsed = (time.monotonic() - t0) / 3600.0
            usage["per_experiment"].append(
                {
                    "name": f"{cond}-seed{seed}",
                    "gpu_hours": elapsed,
                    "train_steps": train_stats["steps"],
                    "last_loss": train_stats["last_loss"],
                    "val_accuracy": eval_stats["accuracy"],
                    "train_label_gold_accuracy": train_acc,
                    "parse_failures": eval_stats["parse_failures"],
                }
            )
            usage["recorded_gpu_hours"] = sum(x.get("gpu_hours", 0.0) for x in usage["per_experiment"])
            (RDIR / "gpu_usage.json").write_text(json.dumps(usage, indent=2) + "\n", encoding="utf-8")
            cleanup_model(model)

    usage["recorded_gpu_hours"] = sum(x.get("gpu_hours", 0.0) for x in usage["per_experiment"])
    (RDIR / "gpu_usage.json").write_text(json.dumps(usage, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {raw_path}")
    print(f"wrote {RDIR / 'gpu_usage.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--student-model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--weak-model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--max-seq-len", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--student-batch-size", type=int, default=4)
    parser.add_argument("--weak-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--smoke-steps", type=int, default=12)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
