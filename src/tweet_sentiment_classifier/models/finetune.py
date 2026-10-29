"""Fine-tune a Hugging Face encoder for 3-class tweet sentiment (PyTorch).

    python -m tweet_sentiment_classifier.models.finetune --model cardiffnlp/twitter-roberta-base-2019-90m \
        --train artifacts/data_transformation/train.csv --val artifacts/data_transformation/val.csv \
        --out artifacts/finetuned/roberta2019-s42 --epochs 3 --seed 42

The best epoch is chosen on the VALIDATION split (macro-F1); the test split is never read here. The saved
checkpoint declares exactly 3 labels (index = sentiment + 1), which is what PredictionPipeline requires, and
can be scored with `evaluation.run --models transformer:<out dir>`.
Check the model id exists first: `python -c "import huggingface_hub as h; print(h.model_info('<id>').id)"`.
"""
import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from tweet_sentiment_classifier.evaluation.metrics import evaluate

ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def _set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batches(n: int, batch_size: int, shuffle: bool, rng: np.random.Generator):
    order = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        yield order[start:start + batch_size]


class TransformerModel:
    """Inference wrapper with the same interface as the other evaluated models."""

    def __init__(self, model_dir: str, max_len: int = 64, batch_size: int = 64, device: str = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.name = f"transformer:{model_dir}"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        if self.model.config.num_labels != 3:
            raise ValueError(f"{model_dir} has {self.model.config.num_labels} labels, expected 3")
        self.max_len, self.batch_size = max_len, batch_size

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        import torch
        texts = list(texts)
        out = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                enc = self.tokenizer(texts[start:start + self.batch_size], truncation=True, padding=True,
                                     max_length=self.max_len, return_tensors="pt").to(self.device)
                out.append(torch.softmax(self.model(**enc).logits, dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 3))

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return self.predict_proba(texts).argmax(1) - 1

    def predict_one(self, text: str) -> int:
        return int(self.predict([text])[0])


def finetune(model_id: str, train_csv: Path, val_csv: Path, out_dir: Path, max_len: int = 64, epochs: int = 3,
             lr: float = 2e-5, batch_size: int = 32, seed: int = 42, train_fraction: float = 1.0,
             warmup_ratio: float = 0.1, device: str = None, progress=print) -> Dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    _set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train = pd.read_csv(train_csv).dropna(subset=["text"])
    if train_fraction < 1.0:
        train = train.groupby("sentiment", group_keys=False).sample(frac=train_fraction, random_state=seed)
    val = pd.read_csv(val_csv).dropna(subset=["text"])
    x_tr, y_tr = train["text"].astype(str).tolist(), train["sentiment"].astype(int).to_numpy() + 1
    x_val, y_val = val["text"].astype(str).tolist(), val["sentiment"].astype(int).to_numpy()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID, ignore_mismatched_sizes=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * int(np.ceil(len(x_tr) / batch_size))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(warmup_ratio * steps), steps)
    rng = np.random.default_rng(seed)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history: List[Dict] = []
    best_f1 = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        start, losses = time.perf_counter(), []
        for idx in _batches(len(x_tr), batch_size, True, rng):
            enc = tokenizer([x_tr[i] for i in idx], truncation=True, padding=True, max_length=max_len,
                            return_tensors="pt").to(device)
            loss = model(**enc, labels=torch.tensor(y_tr[idx], device=device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(float(loss))

        model.eval()
        probs = []
        with torch.no_grad():
            for idx in _batches(len(x_val), 2 * batch_size, False, rng):
                enc = tokenizer([x_val[i] for i in idx], truncation=True, padding=True, max_length=max_len,
                                return_tensors="pt").to(device)
                probs.append(torch.softmax(model(**enc).logits, dim=-1).cpu().numpy())
        proba = np.concatenate(probs)
        m = evaluate(y_val, proba.argmax(1) - 1, proba=proba)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_macro_f1": m["macro_f1"],
                  "val_accuracy": m["accuracy"], "val_ece": m["ece"], "seconds": round(time.perf_counter() - start, 1)}
        history.append(record)
        progress(f"epoch {epoch}: loss {record['train_loss']:.4f}  val macro-F1 {m['macro_f1']:.4f}")
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)

    log = {"model_id": model_id, "seed": seed, "epochs": epochs, "lr": lr, "batch_size": batch_size,
           "max_len": max_len, "train_fraction": train_fraction, "n_train": len(x_tr), "n_val": len(x_val),
           "device": device, "best_val_macro_f1": best_f1,
           "best_epoch": max(history, key=lambda h: h["val_macro_f1"])["epoch"], "history": history,
           "note": "best epoch chosen on the validation split; test split not read"}
    (out_dir / "training_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    return log


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", type=Path, default=Path("artifacts/data_transformation/train.csv"))
    parser.add_argument("--val", type=Path, default=Path("artifacts/data_transformation/val.csv"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=1.0, help="stratified subsample for quick CPU runs")
    args = parser.parse_args(argv)
    log = finetune(args.model, args.train, args.val, args.out, args.max_len, args.epochs, args.lr, args.batch_size,
                   args.seed, args.train_fraction)
    print(f"best val macro-F1 {log['best_val_macro_f1']:.4f} at epoch {log['best_epoch']}; saved to {args.out}")


if __name__ == "__main__":
    main()
