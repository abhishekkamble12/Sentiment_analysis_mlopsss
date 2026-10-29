import json
import os
import re
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tweet_sentiment_classifier import logger
from tweet_sentiment_classifier.config.config_entity import DataTransformationConfig
from tweet_sentiment_classifier.evaluation.metrics import fingerprint


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def clean_text(self, text: str) -> str:
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

    @staticmethod
    def _near_duplicate_key(text: str) -> str:
        return re.sub(r'\s+', ' ', re.sub(r'\d+', '', text)).strip()

    def _split_summary(self, df: pd.DataFrame) -> dict:
        return {
            "n": int(len(df)),
            "label_counts": {str(int(k)): int(v) for k, v in df["sentiment"].value_counts().sort_index().items()},
            "fingerprint": fingerprint(df["text"]),
        }

    def transform_and_split(self):
        """Clean, de-duplicate on the CLEANED text, then split stratified 70/10/20 (train/val/test).

        De-duplicating after cleaning matters: tweets that differ only by case, punctuation or a URL are the
        same text for the model, and would otherwise land on both sides of the split. Texts that appear with
        conflicting labels are dropped as ambiguous.
        """
        try:
            seed = self.config.random_state
            test_size, val_size = self.config.test_size, self.config.val_size
            logger.info(f"Reading dataset from {self.config.data_path}")
            df = pd.read_csv(self.config.data_path)
            n_raw = len(df)

            df["text"] = df["text"].apply(self.clean_text)
            df = df[df["text"] != ""].dropna(subset=["sentiment"])
            n_nonempty = len(df)

            label_count = df.groupby("text")["sentiment"].transform("nunique")
            n_conflict = int((label_count > 1).sum())
            df = df[label_count == 1]
            n_before_dedup = len(df)
            df = df.drop_duplicates(subset="text", keep="first")
            n_duplicates = n_before_dedup - len(df)

            logger.info(f"Splitting {len(df)} rows stratified: test={test_size}, val={val_size}, seed={seed}")
            train_val, test_df = train_test_split(df, test_size=test_size, stratify=df["sentiment"], random_state=seed)
            train_df, val_df = train_test_split(
                train_val, test_size=val_size / (1.0 - test_size), stratify=train_val["sentiment"], random_state=seed
            )
            train_df, val_df, test_df = (d.reset_index(drop=True) for d in (train_df, val_df, test_df))

            sets = {name: set(d["text"]) for name, d in (("train", train_df), ("val", val_df), ("test", test_df))}
            overlaps = {f"{a}_{b}": len(sets[a] & sets[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
            if any(overlaps.values()):
                raise RuntimeError(f"train/val/test share texts after de-duplication: {overlaps}")
            train_keys = {self._near_duplicate_key(t) for t in train_df["text"]}
            near_dup = int(sum(self._near_duplicate_key(t) in train_keys for t in test_df["text"]))

            manifest = {
                "source": str(self.config.data_path),
                "seed": seed, "test_size": test_size, "val_size": val_size,
                "rows": {"raw": n_raw, "empty_after_cleaning_dropped": n_raw - n_nonempty,
                         "conflicting_label_rows_dropped": n_conflict, "duplicates_dropped": n_duplicates,
                         "final": int(len(df))},
                "splits": {"train": self._split_summary(train_df), "val": self._split_summary(val_df),
                           "test": self._split_summary(test_df)},
                "exact_text_overlap": overlaps,
                # informational: test texts equal to a train text once digits are ignored
                "test_near_duplicates_of_train": near_dup,
            }

            for path, frame in ((self.config.train_data_path, train_df), (self.config.val_data_path, val_df),
                                (self.config.test_data_path, test_df)):
                os.makedirs(Path(path).parent, exist_ok=True)
                frame.to_csv(path, index=False)
            Path(self.config.manifest_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            logger.info(f"train/val/test = {len(train_df)}/{len(val_df)}/{len(test_df)}; "
                        f"dropped: {n_raw - n_nonempty} empty, {n_conflict} conflicting, {n_duplicates} duplicates")

        except Exception as e:
            logger.exception(f"Error during data transformation: {e}")
            raise e
