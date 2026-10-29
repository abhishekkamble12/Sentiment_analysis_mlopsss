from tweet_sentiment_classifier.constants import *
from tweet_sentiment_classifier.utils.common import read_yaml, create_directories
from tweet_sentiment_classifier.config.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    BestModelConfig
)
from pathlib import Path

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        return DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
            source_type=config.source_type
        )

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        create_directories([config.root_dir])
        return DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir=config.unzip_data_dir,
            validated_data_path=config.validated_data_file,
            all_schema=self.schema
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([config.root_dir])
        return DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            preprocessed_object_file=config.preprocessed_object_file,
            test_size=config.test_size,
            random_state=config.random_state
        )

    def get_model_trainer_config(self, model_name: str) -> ModelTrainerConfig:
        config = self.config.model_trainer
        data_transform_config = self.config.data_transformation
        
        hyperparams = dict(self.params.get(model_name, {}))
        random_state = int(self.params.get("random_state", 42))
        max_features = int(self.params.get("max_features", 15000))

        create_directories([config.root_dir])

        if model_name == "DistilBERT":
            trained_model_path = Path(config.root_dir) / f"model_{model_name}"
        else:
            trained_model_path = Path(config.root_dir) / f"model_{model_name}.pkl"

        return ModelTrainerConfig(
            root_dir=config.root_dir,
            trained_model_path=trained_model_path,
            train_data_path=data_transform_config.train_data_path,
            test_data_path=data_transform_config.test_data_path,
            model_name=model_name,
            random_state=random_state,
            max_features=max_features,
            hyperparams=hyperparams
        )

    def get_model_evaluation_config(self, model_name: str) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        
        hyperparams = dict(self.params.get(model_name, {}))
        random_state = int(self.params.get("random_state", 42))

        create_directories([config.root_dir])

        if model_name == "DistilBERT":
            model_path = Path(self.config.model_trainer.root_dir) / f"model_{model_name}"
        else:
            model_path = Path(self.config.model_trainer.root_dir) / f"model_{model_name}.pkl"

        return ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            model_path=model_path,
            metrics_file=config.metrics_file,
            model_name=model_name,
            random_state=random_state,
            hyperparams=hyperparams
        )

    def get_best_model_config(self) -> BestModelConfig:
        config = self.config.best_model
        create_directories([config.root_dir])
        
        return BestModelConfig(
            root_dir=config.root_dir,
            model_path=config.model_path,
            metadata_path=config.metadata_path,
            metrics_dir=self.config.model_evaluation.root_dir
        )
