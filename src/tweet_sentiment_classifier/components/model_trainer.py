import os
import joblib
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from tweet_sentiment_classifier import logger
from tweet_sentiment_classifier.config.config_entity import ModelTrainerConfig
class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        try:
            logger.info(f"Loading training dataset for model: {self.config.model_name}")
            train_df = pd.read_csv(self.config.train_data_path)
            
            # Text uses 'text' column based on new schema
            X_train = train_df['text'].fillna("")
            y_train = train_df['sentiment']

            model_path = Path(self.config.trained_model_path)
            os.makedirs(model_path.parent, exist_ok=True)
            
            model_name = self.config.model_name
            hyperparams = self.config.hyperparams
            random_state = self.config.random_state
            
            if model_name in ["LogisticRegression", "LinearSVC", "MultinomialNB", "RidgeClassifier", "XGBoost"]:
                logger.info(f"Vectorizing features using TF-IDF (max_features={self.config.max_features})")
                vectorizer = TfidfVectorizer(max_features=self.config.max_features, ngram_range=(1, 2))
                X_train_vectorized = vectorizer.fit_transform(X_train)
                
                # y_train for XGBoost needs to be 0, 1, 2
                y_train_mapped = (y_train + 1).astype(int).values

                if model_name == "LogisticRegression":
                    classifier = LogisticRegression(
                        C=hyperparams.get("C", 1.0),
                        solver=hyperparams.get("solver", "lbfgs"),
                        max_iter=hyperparams.get("max_iter", 100),
                        random_state=random_state,
                        n_jobs=-1
                    )
                elif model_name == "LinearSVC":
                    classifier = LinearSVC(
                        C=hyperparams.get("C", 1.0),
                        loss=hyperparams.get("loss", "squared_hinge"),
                        max_iter=hyperparams.get("max_iter", 1000),
                        random_state=random_state
                    )
                elif model_name == "MultinomialNB":
                    classifier = MultinomialNB(alpha=hyperparams.get("alpha", 0.1))
                elif model_name == "RidgeClassifier":
                    classifier = RidgeClassifier(
                        alpha=hyperparams.get("alpha", 1.0),
                        random_state=random_state
                    )
                elif model_name == "XGBoost":
                    from xgboost import XGBClassifier
                    classifier = XGBClassifier(
                        n_estimators=hyperparams.get("n_estimators", 80),
                        max_depth=hyperparams.get("max_depth", 5),
                        learning_rate=hyperparams.get("learning_rate", 0.15),
                        random_state=random_state,
                        n_jobs=-1,
                        eval_metric='mlogloss'
                    )
                    # XGBoost natively maps labels
                    y_train = y_train_mapped

                logger.info(f"Training {model_name}...")
                classifier.fit(X_train_vectorized, y_train)
                
                joblib.dump(classifier, model_path)
                
                preprocessor_path = Path("artifacts/data_transformation/preprocessor.pkl")
                os.makedirs(preprocessor_path.parent, exist_ok=True)
                joblib.dump(vectorizer, preprocessor_path)
                
                logger.info(f"Saved model and vectorizer for {model_name}")
                
            elif model_name == "DistilBERT":
                import os as env_os
                env_os.environ["TF_USE_LEGACY_KERAS"] = "1"
                import tensorflow as tf
                from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

                logger.info("Initializing DistilBERT training (CPU Optimized)")
                
                # Retrieve sample size from hyperparams or default to 500 for better quality
                sample_size_param = hyperparams.get("sample_size", 500)
                sample_size = min(sample_size_param, len(train_df))
                logger.info(f"Sampling {sample_size} records for DistilBERT safely under hardware constraints")
                sampled_train_df = train_df.sample(n=sample_size, random_state=random_state)
                
                X_train_str = sampled_train_df['text'].astype(str).tolist()
                y_train_mapped_sampled = (sampled_train_df['sentiment'] + 1).astype(int).values

                tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_safetensors=False)
                model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3, use_safetensors=False)

                # Freeze the DistilBERT backbone (Feature Extraction mode) for low-cost, fast training
                logger.info("Freezing DistilBERT base model layers for high-efficiency feature extraction...")
                model.layers[0].trainable = False

                train_encodings = tokenizer(X_train_str, truncation=True, padding=True, max_length=hyperparams.get("max_len", 128), return_tensors="tf")
                
                train_dataset = tf.data.Dataset.from_tensor_slices((
                    dict(train_encodings),
                    y_train_mapped_sampled
                )).shuffle(50).batch(hyperparams.get("batch_size", 32))

                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparams.get("learning_rate", 2e-5)),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy']
                )

                model.fit(train_dataset, epochs=hyperparams.get("epochs", 5))

                model.save_pretrained(str(model_path))
                tokenizer.save_pretrained(str(model_path))
                logger.info(f"Saved DistilBERT model and tokenizer at {model_path}")
            
            else:
                raise ValueError(f"Unknown model_name: {model_name}")

        except Exception as e:
            logger.exception(f"Error in ModelTrainer: {e}")
            raise e
