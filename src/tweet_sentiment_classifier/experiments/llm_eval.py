"""Zero-shot LLM baseline with a response cache (reruns cost nothing).

    python -m tweet_sentiment_classifier.experiments.llm_eval --data data/gold/gold_test.csv \
        --out artifacts/experiments/llm/groq-llama-3.1-8b --model llama-3.1-8b-instant

Needs GROQ_API_KEY. Uses the hardened prompt from utils/llm_safety.py at temperature 0. Every raw reply is
stored in cache.jsonl. Unparseable replies are NOT given a label: they are reported as invalid, and scored
two ways: metrics on valid replies only (with coverage), and strict accuracy counting invalid as wrong.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from tweet_sentiment_classifier.evaluation.metrics import bootstrap_ci, evaluate
from tweet_sentiment_classifier.utils.llm_safety import build_sentiment_prompt, parse_sentiment_json


def groq_completer(model: str, api_key: str) -> Callable[[str], str]:
    from langchain_core.messages import HumanMessage
    from langchain_groq import ChatGroq
    llm = ChatGroq(model_name=model, api_key=api_key, temperature=0.0)
    return lambda prompt: llm.invoke([HumanMessage(content=prompt)]).content


class CachedLLMClassifier:
    def __init__(self, complete: Callable[[str], str], model: str, cache_path: Path, max_retries: int = 5,
                 backoff_seconds: float = 2.0, sleep: Callable[[float], None] = time.sleep):
        self.complete, self.model = complete, model
        self.cache_path = Path(cache_path)
        self.max_retries, self.backoff, self.sleep = max_retries, backoff_seconds, sleep
        self.cache: Dict[str, str] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.cache[row["key"]] = row["response"]
        self.api_calls = 0

    def _key(self, prompt: str) -> str:
        return hashlib.sha256(f"{self.model}\n{prompt}".encode("utf-8")).hexdigest()

    def raw(self, text: str) -> Optional[str]:
        prompt = build_sentiment_prompt(text)
        key = self._key(prompt)
        if key in self.cache:
            return self.cache[key]
        for attempt in range(self.max_retries):
            try:
                self.api_calls += 1
                response = self.complete(prompt)
                break
            except Exception:  # rate limits and transient errors: back off and retry
                if attempt == self.max_retries - 1:
                    return None
                self.sleep(self.backoff * (2 ** attempt))
        self.cache[key] = response
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "model": self.model, "text": text, "response": response}) + "\n")
        return response

    def classify(self, text: str) -> Optional[int]:
        parsed = parse_sentiment_json(self.raw(text))
        return None if parsed is None else parsed[0]


def evaluate_llm(data_csv: Path, out_dir: Path, classifier: CachedLLMClassifier, weights_column: str = None) -> Dict:
    df = pd.read_csv(data_csv).dropna(subset=["text"])
    y = df["sentiment"].astype(int).to_numpy()
    preds = [classifier.classify(t) for t in df["text"].astype(str)]
    valid = np.array([p is not None for p in preds])
    pred_valid = np.array([p for p in preds if p is not None], dtype=int)
    w = df[weights_column].astype(float).to_numpy() if weights_column else None

    result = {"model": classifier.model, "n": int(len(df)), "n_valid": int(valid.sum()),
              "coverage": float(valid.mean()) if len(df) else 0.0, "api_calls": classifier.api_calls,
              "strict_accuracy": float(np.mean([p == t for p, t in zip(preds, y)])) if len(df) else 0.0}
    if valid.any():
        wv = None if w is None else w[valid]
        m = evaluate(y[valid], pred_valid, sample_weight=wv)
        m["macro_f1_ci"] = bootstrap_ci(y[valid], pred_valid, sample_weight=wv)
        result["valid_only"] = m
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame({"text": df["text"], "label": y, "pred": [("" if p is None else p) for p in preds]}).to_csv(
        out_dir / "predictions.csv", index=False)
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="llama-3.1-8b-instant")
    parser.add_argument("--weights-column", default=None)
    args = parser.parse_args(argv)
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        parser.error("GROQ_API_KEY is not set")
    clf = CachedLLMClassifier(groq_completer(args.model, key), args.model, args.out / "cache.jsonl")
    result = evaluate_llm(args.data, args.out, clf, args.weights_column)
    print(json.dumps({k: v for k, v in result.items() if k != "valid_only"}, indent=2))
    if "valid_only" in result:
        print(f"macro-F1 on valid replies: {result['valid_only']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
