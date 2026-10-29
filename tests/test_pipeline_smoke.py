"""Fresh-clone smoke test: ingest -> validate -> transform -> train, using the repo's real config files (audit B9)."""
import random
import shutil
from pathlib import Path

import joblib
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]

WORDS = {
    -1: "terrible awful corrupt failure angry disaster worst hate".split(),
    0: "meeting report today scheduled announced minister city date".split(),
    1: "great excellent happy success proud wonderful best love".split(),
}


def make_dataset(path: Path, n=300):
    rng = random.Random(0)
    rows = []
    for i in range(n):
        label = (-1, 0, 1)[i % 3]
        words = rng.sample(WORDS[label], 4) + [f"tok{i}"]  # tok{i} keeps every text unique
        rows.append({"text": " ".join(words), "sentiment": float(label), "cleaned_text_final": "x"})
    df = pd.DataFrame(rows)
    # Rows validation must remove: a null text, a blank text, and an exact duplicate row
    junk = pd.DataFrame([
        {"text": None, "sentiment": 0.0, "cleaned_text_final": "x"},
        {"text": "   ", "sentiment": 1.0, "cleaned_text_final": "x"},
        df.iloc[0].to_dict(),
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([df, junk], ignore_index=True).to_csv(path, index=False)
    return n


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def real_config_manager():
    from tweet_sentiment_classifier.config.configuration import ConfigurationManager
    return ConfigurationManager(
        config_filepath=REPO / "config" / "config.yaml",
        params_filepath=REPO / "params.yaml",
        schema_filepath=REPO / "schema.yaml",
    )


def test_configured_source_dataset_exists_in_the_repo():
    """The shipped config must point at a file that is actually in the repository."""
    from tweet_sentiment_classifier.utils.common import read_yaml
    cfg = read_yaml(REPO / "config" / "config.yaml")
    assert (REPO / cfg.data_ingestion.source_URL).is_file(), cfg.data_ingestion.source_URL


def test_pipeline_runs_from_a_clean_directory(workdir):
    from tweet_sentiment_classifier.components.data_ingestion import DataIngestion
    from tweet_sentiment_classifier.components.data_validation import DataValidation
    from tweet_sentiment_classifier.components.data_transformation import DataTransformation
    from tweet_sentiment_classifier.components.model_trainer import ModelTrainer

    cm = real_config_manager()
    source = workdir / cm.config.data_ingestion.source_URL
    n_good = make_dataset(source)

    # --- ingestion copies the source; it does not need network or committed artifacts
    ing = DataIngestion(cm.get_data_ingestion_config())
    ingested = ing.download_file()
    ing.extract_zip_file(ingested)
    assert ingested.is_file()
    ingested_bytes = ingested.read_bytes()

    # --- validation drops the junk and writes a NEW file, leaving its input untouched
    assert DataValidation(cm.get_data_validation_config()).validate_all_data() is True
    assert ingested.read_bytes() == ingested_bytes
    validated = Path(cm.config.data_validation.validated_data_file)
    assert len(pd.read_csv(validated)) == n_good

    # --- transform reads the validated file and produces disjoint train/test
    DataTransformation(cm.get_data_transformation_config()).transform_and_split()
    train = pd.read_csv(cm.config.data_transformation.train_data_path)
    val = pd.read_csv(cm.config.data_transformation.val_data_path)
    test = pd.read_csv(cm.config.data_transformation.test_data_path)
    assert len(train) + len(val) + len(test) == n_good
    assert not set(train["text"]) & set(test["text"])
    assert not set(train["text"]) & set(val["text"]) and not set(val["text"]) & set(test["text"])
    assert Path(cm.config.data_transformation.manifest_path).is_file()

    # --- train a model that the serving code can use
    trainer_cfg = cm.get_model_trainer_config("LogisticRegression")
    ModelTrainer(trainer_cfg).train()
    model = joblib.load(trainer_cfg.trained_model_path)
    vectorizer = joblib.load(workdir / "artifacts" / "data_transformation" / "preprocessor.pkl")
    pred = model.predict(vectorizer.transform(["great excellent happy success"]))[0]
    assert pred == 1


def test_validation_is_idempotent_and_drops_stale_output_on_failure(workdir):
    from tweet_sentiment_classifier.components.data_ingestion import DataIngestion
    from tweet_sentiment_classifier.components.data_validation import DataValidation

    cm = real_config_manager()
    make_dataset(workdir / cm.config.data_ingestion.source_URL)
    DataIngestion(cm.get_data_ingestion_config()).download_file()
    validated = Path(cm.config.data_validation.validated_data_file)

    assert DataValidation(cm.get_data_validation_config()).validate_all_data()
    first = validated.read_bytes()
    assert DataValidation(cm.get_data_validation_config()).validate_all_data()
    assert validated.read_bytes() == first  # re-running does not keep shrinking the data

    # Corrupt the input so validation fails: the previous validated.csv must not survive
    ingested = Path(cm.config.data_validation.unzip_data_dir)
    df = pd.read_csv(ingested)
    df.loc[0, "sentiment"] = 5.0
    df.to_csv(ingested, index=False)
    assert DataValidation(cm.get_data_validation_config()).validate_all_data() is False
    assert not validated.exists()
