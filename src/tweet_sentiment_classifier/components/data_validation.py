import pandas as pd
from pathlib import Path
from tweet_sentiment_classifier import logger
from tweet_sentiment_classifier.config.config_entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_data(self) -> bool:
        try:
            validation_status = True
            logger.info("Starting Data Validation...")

            # A failed run must not leave a previous run's output for the next stage to pick up
            Path(self.config.validated_data_path).unlink(missing_ok=True)

            if not Path(self.config.unzip_data_dir).exists():
                logger.error(f"Data file not found at {self.config.unzip_data_dir}")
                self._write_status("Validation Failed: Data file not found")
                return False

            df = pd.read_csv(self.config.unzip_data_dir)
            
            # 1. Check schema mismatch
            expected_cols = [f["name"] for f in self.config.all_schema["features"]] + [self.config.all_schema["target"]["name"]]
            missing_cols = [col for col in expected_cols if col not in df.columns]
            
            if missing_cols:
                validation_status = False
                msg = f"Validation Failed: Missing columns: {missing_cols}"
                logger.error(msg)
                self._write_status(msg)
                return validation_status

            # 2. Check nulls in text
            text_col = self.config.all_schema["features"][0]["name"]
            if df[text_col].isnull().sum() > 0:
                logger.warning(f"Found {df[text_col].isnull().sum()} nulls in {text_col} column. Dropping them.")
                df = df.dropna(subset=[text_col])

            # 3. Check empty strings
            empty_mask = df[text_col].str.strip() == ""
            if empty_mask.sum() > 0:
                logger.warning(f"Found {empty_mask.sum()} empty strings. Dropping them.")
                df = df[~empty_mask]

            # 4. Check target labels
            target_col = self.config.all_schema["target"]["name"]
            valid_classes = self.config.all_schema["target"]["classes"]
            
            invalid_labels = df[~df[target_col].isin(valid_classes)]
            if len(invalid_labels) > 0:
                validation_status = False
                msg = f"Validation Failed: Invalid labels found in {target_col}: {invalid_labels[target_col].unique()}"
                logger.error(msg)
                self._write_status(msg)
                return validation_status

            # 5. Drop duplicates
            initial_len = len(df)
            df = df.drop_duplicates()
            if len(df) < initial_len:
                logger.info(f"Dropped {initial_len - len(df)} duplicate rows.")

            if validation_status:
                msg = "Validation Passed Successfully"
                logger.info(msg)
                self._write_status(msg)
                # Write the cleaned data to its own file; the ingested input is never modified
                Path(self.config.validated_data_path).parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(self.config.validated_data_path, index=False)

            return validation_status

        except Exception as e:
            logger.exception(f"Error occurred during Data Validation: {str(e)}")
            self._write_status(f"Validation Failed with Error: {str(e)}")
            return False

    def _write_status(self, status: str):
        with open(self.config.STATUS_FILE, 'w') as f:
            f.write(status)
