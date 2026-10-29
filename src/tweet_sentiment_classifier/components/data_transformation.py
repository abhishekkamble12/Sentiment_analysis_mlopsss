import json
import os
import re
import pandas as pd
from pathlib import Path
from typing import Callable, Dict, Tuple
from sklearn.model_selection import train_test_split
from tweet_sentiment_classifier import logger
from tweet_sentiment_classifier.config.config_entity import DataTransformationConfig
from tweet_sentiment_classifier.evaluation.metrics import fingerprint


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _near_duplicate_key(text: str) -> str:
    return re.sub(r'\s+', ' ', re.sub(r'\d+', '', text)).strip()


def _split_summary(df: pd.DataFrame) -> dict:
    return {
        "n": int(len(df)),
        "label_counts": {str(int(k)): int(v) for k, v in df["sentiment"].value_counts().sort_index().items()},
        "fingerprint": fingerprint(df["text"]),
    }


def clean_and_split(raw: pd.DataFrame, test_size: float, val_size: float, seed: int,
                    clean_fn: Callable[[str], str] = clean_text) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Clean, de-duplicate on the CLEANED text, then split stratified (train/val/test). Pure: writes nothing.

    De-duplicating after cleaning matters: tweets that differ only by case, punctuation or a URL are the
    same text for the model, and would otherwise land on both sides of the split. Texts that appear with
    conflicting labels are dropped as ambiguous. Returns (train, val, test, manifest).
    """
    df = raw.copy()
    n_raw = len(df)
    df["text"] = df["text"].apply(clean_fn)
    df = df[df["text"] != ""].dropna(subset=["sentiment"])
    n_nonempty = len(df)

    label_count = df.groupby("text")["sentiment"].transform("nunique")
    n_conflict = int((label_count > 1).sum())
    df = df[label_count == 1]
    n_before_dedup = len(df)
    df = df.drop_duplicates(subset="text", keep="first")
    n_duplicates = n_before_dedup - len(df)

    train_val, test_df = train_test_split(df, test_size=test_size, stratify=df["sentiment"], random_state=seed)
    train_df, val_df = train_test_split(
        train_val, test_size=val_size / (1.0 - test_size), stratify=train_val["sentiment"], random_state=seed
    )
    train_df, val_df, test_df = (d.reset_index(drop=True) for d in (train_df, val_df, test_df))

    sets = {name: set(d["text"]) for name, d in (("train", train_df), ("val", val_df), ("test", test_df))}
    overlaps = {f"{a}_{b}": len(sets[a] & sets[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    if any(overlaps.values()):
        raise RuntimeError(f"train/val/test share texts after de-duplication: {overlaps}")
    train_keys = {_near_duplicate_key(t) for t in train_df["text"]}
    near_dup = int(sum(_near_duplicate_key(t) in train_keys for t in test_df["text"]))

    manifest = {
        "seed": seed, "test_size": test_size, "val_size": val_size,
        "rows": {"raw": n_raw, "empty_after_cleaning_dropped": n_raw - n_nonempty,
                 "conflicting_label_rows_dropped": n_conflict, "duplicates_dropped": n_duplicates,
                 "final": int(len(df))},
        "splits": {"train": _split_summary(train_df), "val": _split_summary(val_df), "test": _split_summary(test_df)},
        "exact_text_overlap": overlaps,
        # informational: test texts equal to a train text once digits are ignored
        "test_near_duplicates_of_train": near_dup,
    }
    return train_df, val_df, test_df, manifest


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def clean_text(self, text: str) -> str:
        return clean_text(text)

    def transform_and_split(self):
        try:
            logger.info(f"Reading dataset from {self.config.data_path}")
            raw = pd.read_csv(self.config.data_path)
            train_df, val_df, test_df, manifest = clean_and_split(
                raw, self.config.test_size, self.config.val_size, self.config.random_state, self.clean_text
            )
            manifest["source"] = str(self.config.data_path)

            for path, frame in ((self.config.train_data_path, train_df), (self.config.val_data_path, val_df),
                                (self.config.test_data_path, test_df)):
                os.makedirs(Path(path).parent, exist_ok=True)
                frame.to_csv(path, index=False)
            Path(self.config.manifest_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            rows = manifest["rows"]
            logger.info(f"train/val/test = {len(train_df)}/{len(val_df)}/{len(test_df)}; dropped: "
                        f"{rows['empty_after_cleaning_dropped']} empty, {rows['conflicting_label_rows_dropped']} "
                        f"conflicting, {rows['duplicates_dropped']} duplicates")

        except Exception as e:
            logger.exception(f"Error during data transformation: {e}")
            raise e
