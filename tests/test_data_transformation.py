"""The split must be leak-free by construction and fully described by its manifest (audit B7)."""
import json
from pathlib import Path

import pandas as pd
import pytest

from tweet_sentiment_classifier.components.data_transformation import DataTransformation
from tweet_sentiment_classifier.config.config_entity import DataTransformationConfig
from tweet_sentiment_classifier.evaluation.metrics import fingerprint


def make_config(tmp_path, data_path, seed=42, test_size=0.2, val_size=0.1):
    return DataTransformationConfig(
        root_dir=tmp_path, data_path=data_path, train_data_path=tmp_path / "train.csv",
        test_data_path=tmp_path / "test.csv", preprocessed_object_file=tmp_path / "p.pkl",
        test_size=test_size, random_state=seed, val_data_path=tmp_path / "val.csv",
        manifest_path=tmp_path / "manifest.json", val_size=val_size)


def write_source(path, rows):
    pd.DataFrame(rows, columns=["text", "sentiment"]).to_csv(path, index=False)


def base_rows(n_per_class=100):
    rows = []
    for label, word in ((-1, "awful"), (0, "report"), (1, "great")):
        rows += [(f"{word} item number {i} about topic", float(label)) for i in range(n_per_class)]
    return rows


def run(tmp_path, rows, **kw):
    src = tmp_path / "src.csv"
    write_source(src, rows)
    cfg = make_config(tmp_path, src, **kw)
    DataTransformation(cfg).transform_and_split()
    read = lambda name: pd.read_csv(tmp_path / name)  # noqa: E731
    return read("train.csv"), read("val.csv"), read("test.csv"), json.loads((tmp_path / "manifest.json").read_text())


def test_sizes_are_70_10_20_and_stratified(tmp_path):
    train, val, test, m = run(tmp_path, base_rows())
    assert (len(train), len(val), len(test)) == (210, 30, 60)
    for frame in (train, val, test):
        counts = frame["sentiment"].value_counts(normalize=True)
        assert all(abs(share - 1 / 3) < 0.02 for share in counts)       # class balance preserved in every split
    assert m["splits"]["train"]["n"] == 210 and m["rows"]["final"] == 300


def test_splits_are_disjoint(tmp_path):
    train, val, test, m = run(tmp_path, base_rows())
    texts = [set(f["text"]) for f in (train, val, test)]
    assert not (texts[0] & texts[1]) and not (texts[0] & texts[2]) and not (texts[1] & texts[2])
    assert m["exact_text_overlap"] == {"train_val": 0, "train_test": 0, "val_test": 0}


def test_texts_that_only_differ_after_cleaning_cannot_straddle_the_split(tmp_path):
    # Old behaviour: dedup ran on the raw text, so these survived as "different" and leaked across splits.
    variants = ["Great news today!", "great news today", "GREAT NEWS TODAY?!", "great news today https://t.co/abc"]
    rows = base_rows(60) + [(v, 1.0) for v in variants]
    train, val, test, m = run(tmp_path, rows)
    everywhere = pd.concat([train, val, test])
    assert (everywhere["text"] == "great news today").sum() == 1
    assert m["rows"]["duplicates_dropped"] == 3


def test_texts_with_conflicting_labels_are_dropped_entirely(tmp_path):
    rows = base_rows(60) + [("same words here", 1.0), ("Same words here!", -1.0), ("same words here", 0.0)]
    train, val, test, m = run(tmp_path, rows)
    assert "same words here" not in set(pd.concat([train, val, test])["text"])
    assert m["rows"]["conflicting_label_rows_dropped"] == 3


def test_rows_that_are_empty_after_cleaning_are_dropped(tmp_path):
    rows = base_rows(60) + [("!!! ???", 1.0), ("https://t.co/x", 0.0)]
    *_, m = run(tmp_path, rows)
    assert m["rows"]["empty_after_cleaning_dropped"] == 2 and m["rows"]["final"] == 180


def test_manifest_fingerprints_match_the_written_files_and_are_reproducible(tmp_path):
    train, val, test, m = run(tmp_path, base_rows())
    assert m["splits"]["test"]["fingerprint"] == fingerprint(test["text"])
    again = tmp_path / "again"
    again.mkdir()
    *_, m2 = run(again, base_rows())
    assert m2["splits"] == m["splits"]                                  # same seed -> identical splits


def test_a_different_seed_gives_a_different_split(tmp_path):
    *_, m1 = run(tmp_path, base_rows(), seed=1)
    other = tmp_path / "other"
    other.mkdir()
    *_, m2 = run(other, base_rows(), seed=2)
    assert m1["splits"]["test"]["fingerprint"] != m2["splits"]["test"]["fingerprint"]


def test_near_duplicates_that_differ_only_by_digits_are_reported_not_hidden(tmp_path):
    rows = base_rows(60) + [(f"vote for modi in {year}", 1.0) for year in range(2010, 2030)]
    *_, m = run(tmp_path, rows)
    assert m["test_near_duplicates_of_train"] >= 1        # exact dedup cannot catch these; the manifest says so
