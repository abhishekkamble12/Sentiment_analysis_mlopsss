# Tweet Sentiment Classification

Three-class (negative / neutral / positive) sentiment classification for tweets, with a training
pipeline, a FastAPI service and a small web UI. Work in progress: see *Known limitations* before
quoting any number from this repository.

## Known limitations

- **The dataset labels are not human annotations.** `sentiment_analysis/dataset/thoroughly_cleaned_data.csv`
  (163k tweets, ~91% mention "modi") was labelled by a lexicon tool: TextBlob's `sign(polarity)` agrees
  with its labels on 99.98% of a 20,000-row random sample. Every accuracy / F1 in `artifacts/` therefore
  measures how well a model imitates that lexicon, **not** how well it detects human-perceived sentiment.
  A human-labelled evaluation set is planned.
- Use `artifacts/evaluation/results.csv` (one run, one leak-free split, with confidence intervals; see
  *Evaluation*). The older `artifacts/model_evaluation/` files mix results from different runs and splits
  and are superseded. The deployed `artifacts/model_trainer/*.pkl` models were trained on the older, leaky split.
- The train/val/test CSVs are generated, not committed: `python main.py` recreates them deterministically, and
  `artifacts/data_transformation/split_manifest.json` holds their sizes, label counts and fingerprints to verify them.
- The transformer tier is disabled until a fine-tuned 3-class checkpoint exists (see below).

## How a prediction is made

`INFERENCE_MODE` (environment variable, default `local`) controls who decides:

| Mode | Behaviour |
|---|---|
| `local` | The trained scikit-learn model decides. No external API is called. |
| `llm` | An LLM (Hugging Face, then Groq) decides; if both fail the local model answers. |
| `hybrid` | The local model decides unless its confidence is below `HYBRID_CONFIDENCE_THRESHOLD` (0.75), then the LLM decides. |

Every response reports `decision_source` (for example `local:logisticregression` or
`llm:groq/llama-3.1-8b-instant`), the vote of each model that ran, and any `warnings`. LLM decisions have
no confidence (`null`). The optional transformer tier (`artifacts/model_trainer/model_DistilBERT`) is loaded
only if its `config.json` declares exactly 3 labels; it contributes a vote but does not decide.

## Setup

Python 3.12. Dependencies are split by purpose in `requirements/`:

| File | For |
|---|---|
| `base.txt` | data handling, config, database |
| `api.txt` | web app (no ML frameworks) |
| `serve.txt` | production serving: adds PyTorch, transformers, LangChain, Postgres driver |
| `train.txt` | the training pipeline (TensorFlow, MLflow, XGBoost). Install in a **separate** environment from `serve.txt` |
| `eval.txt` | lexicon baselines (TextBlob, VADER) for the evaluation runner |
| `dev.txt` | tests and lint |

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                         # = requirements/serve.txt
cp .env.example .env                                    # then edit; all keys are optional in local mode
python app.py                                           # http://127.0.0.1:8000
```

`.env` keys: `DATABASE_URL` (default `sqlite:///./monitoring.db`), `INFERENCE_MODE`, `GROQ_API_KEY`,
`HF_TOKEN`. Without LangChain/`GROQ_API_KEY` the app still starts; `/search-analysis` then returns 503.

Existing databases created by older versions need the migration:

```bash
alembic upgrade head
```

## Training pipeline

```bash
pip install -r requirements/train.txt        # in its own environment
python main.py
```

Stages: ingest, validate (writes `artifacts/data_validation/validated.csv`), transform and split, then
train and evaluate each model in `params.yaml` (MLflow logging) and pick the best by macro-F1.

## Docker

```bash
export POSTGRES_PASSWORD='choose-one'         # required; compose refuses to start without it
docker compose up --build
```

The database listens on `127.0.0.1:5432` only. The image installs the CPU-only PyTorch build.

## Evaluation

One command scores every model on one test set, so all numbers share a run id, a test-set fingerprint and
a seed (models are fit from scratch; nothing is read from old pickles):

```bash
pip install -r requirements/base.txt -r requirements/eval.txt
python -m tweet_sentiment_classifier.evaluation.run --train artifacts/data_transformation/train.csv --test artifacts/data_transformation/test.csv --out artifacts/evaluation
```

It writes `results.csv` / `results.json` (rebuilt from scratch each run) with accuracy, macro-F1 with a
95% bootstrap CI, Cohen's kappa, expected calibration error, per-class F1 and per-text latency, and reports
how many test texts also occur in the training file. Baselines: majority class, TextBlob, VADER.
**Read TextBlob's score as a check on the labels, not as a competitor**: it generated them.

Current silver results (`artifacts/evaluation/results.csv`, 31,917 test tweets, 0% train/test text overlap,
hyperparameters tuned on the validation split, see `tuning.json`):

| Model | Macro-F1 [95% CI] |
|---|---|
| TextBlob (label source) | 0.9997 [0.9994, 0.9998] |
| LinearSVC (served) | 0.9347 [0.9319, 0.9372] |
| LogisticRegression | 0.9262 [0.9231, 0.9288] |
| RidgeClassifier | 0.9014 [0.8981, 0.9046] |
| MultinomialNB | 0.7448 [0.7396, 0.7495] |
| VADER | 0.5518 [0.5463, 0.5568] |
| Majority | 0.2053 [0.2035, 0.2069] |

Over 5 different random splits (`seed_study.csv`) the standard deviation of macro-F1 is 0.002 to 0.003 for
every trained model, so gaps larger than about 0.01 are not split noise. `comparisons.csv` has paired-bootstrap
and McNemar tests (Holm-adjusted) for every pair; the bootstrap p-values are floored at 1/1001, so their
Holm-adjusted floor is 0.015. These are **silver** results: agreement with TextBlob, not with people. The
`/results` page shows them next to the gold results once the human-labelled set exists.

### Research workflow (gold set and experiments)

```bash
python -m tweet_sentiment_classifier.gold.sample --pilot                 # 1. pilot sample for annotators
python -m tweet_sentiment_classifier.gold.agreement data/gold/ann_a.csv data/gold/ann_b.csv   # 2. alpha
python -m tweet_sentiment_classifier.gold.sample --exclude data/gold/pilot_private.csv         # 3. full sample
python -m tweet_sentiment_classifier.gold.adjudicate --primary data/gold/ann_a.csv data/gold/ann_b.csv --tiebreak data/gold/ann_c.csv
python -m tweet_sentiment_classifier.gold.freeze                          # 4. gold_test / gold_train_pool
python -m tweet_sentiment_classifier.evaluation.run --train artifacts/data_transformation/train.csv --test data/gold/gold_test.csv --out artifacts/evaluation/gold --label-source gold --weights-column sampling_weight
python -m tweet_sentiment_classifier.experiments.gold_budget             # 5. experiments (see each module's --help)
python -m tweet_sentiment_classifier.experiments.cleanlab_run --eval gold=data/gold/gold_test.csv
python -m tweet_sentiment_classifier.experiments.report --gold artifacts/evaluation/gold/results.json --budget artifacts/experiments/gold_budget/results.json
```

Annotators follow `docs/annotation_guidelines.md` (draft; review it before the pilot). Fine-tuning
(`models/finetune.py`) needs a GPU for the full training split; LLM evaluation needs `GROQ_API_KEY`.

## Tests

```bash
pip install -r requirements/dev.txt
python -m pytest
```

CI runs the same command. `tests/test_transformer_inference.py` additionally needs the serving
environment (`requirements/serve.txt`) and is skipped otherwise.
