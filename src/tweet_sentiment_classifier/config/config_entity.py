from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    source_type: str

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    validated_data_path: Path
    all_schema: dict

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    train_data_path: Path
    test_data_path: Path
    preprocessed_object_file: Path
    test_size: float
    random_state: int

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    trained_model_path: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    random_state: int
    max_features: int
    hyperparams: Dict[str, Any]

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    metrics_file: Path
    model_name: str
    random_state: int
    hyperparams: Dict[str, Any]

@dataclass(frozen=True)
class BestModelConfig:
    root_dir: Path
    model_path: Path
    metadata_path: Path
    metrics_dir: Path
