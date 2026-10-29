# Remaining work: execution plan for an autonomous coding agent

Audience: a small coding model (Gemini Flash-8B class) working in this repository with file-editing and shell
tools, supervised by a human. Everything below is written to be followed literally. Do one work package (WP)
at a time, in order, and commit after each one.

---------------------------------------------------------------------------------------------------------

## Status update (supersedes section 3 where they differ)

The code for WP1 to WP6 has been written, tested and committed (branches `feat/tuning-significance`,
`feat/gold-tooling`, `feat/experiments`, `feat/demo-and-db`). Done and measured: tuning (LR C=4, LinearSVC C=1,
NB alpha=0.01, Ridge alpha=1), final silver table, comparisons, 5-seed study (seed 42 reproduces the committed
test fingerprint), retrained served models (LinearSVC selected). What remains needs a human or resources:

1. Review `docs/annotation_guidelines.md`, run the pilot, have 2 to 3 people annotate, then sample, adjudicate,
   freeze (commands in README "Research workflow").
2. With the frozen gold set: gold evaluation, gold budget, confident learning, error sample, behavioural-case
   review (WP4 code exists; run it and record outputs).
3. GPU: `models/finetune.py` on the full split (3 seeds per model; verify model ids first). Key: `llm_eval.py`.
4. Docker / Postgres verification (WP5 steps 2 and 3), pushing branches, venue search, paper (WP7).

Treat any WP item below as "verify it still holds" rather than "build it".

## 0. Read this first: the rules that override everything else

1. **Never invent results.** A number, a table cell, a citation, a DOI or a "finding" must come from a command
   you actually ran in this session or from a source you actually opened. If you could not run it, write
   "NOT RUN" and say why. Never write expected or typical values.
2. **Never touch the test split to make a choice.** Hyperparameters, thresholds, feature settings and model
   choice are decided on the validation split (silver) or on `gold_train_pool` (gold). Test files
   (`test.csv`, `gold_test.csv`) are read only by the final evaluation commands.
3. **Never create labels.** You must not label tweets yourself or generate "synthetic annotations" for the gold
   set. Human annotators do that (WP3). If a task needs labels that do not exist, stop and ask.
4. **One WP = one branch off the previous one = small commits.** Do not start the next WP until the current
   one's *Acceptance* checklist is fully ticked and the full test suite passes.
5. **Read before you edit.** Open the file, find the exact text, change only that. Do not rewrite a whole file
   unless it is under ~80 lines or you created it.
6. **Tests first, and prove they can fail.** For every behaviour you add: write the test, see it pass, then
   break the code on purpose (a "mutation"), see the test fail, restore the code. If breaking the code does not
   fail any test, the test is too weak; fix the test.
7. **Do not weaken a test to make it pass.** Fix the code or fix the test's *setup* (and say which).
8. **Ask the human instead of guessing** for anything in section 6 ("Needs a human").
9. **Destructive actions need explicit approval**: deleting tracked files, `git push`, `git reset --hard`,
   force operations, dropping database tables, deleting `artifacts/`. Default is: do not.
10. **Report honestly after every WP** using the template in section 7.

Stop and report (do not try a third time) when: the same command fails twice with the same error; the test
count goes down; a number you reproduce differs from the "known facts" in section 3; a command wants network
access or a key you do not have.

---------------------------------------------------------------------------------------------------------

## 1. Environment (Windows 11) and tool gotchas

- Repo root: `C:\Users\Abhishek\Sentiment_analysis_mlopsss`. Python 3.12. Shells: PowerShell and Git Bash.
- **Create virtual environments inside the repo folder or at a SHORT path.** Windows has a 260-character path
  limit and installing PyTorch into a deep path fails. Use `C:\sv` for the PyTorch environment.

  ```powershell
  python -m venv .venv-dev
  .\.venv-dev\Scripts\python -m pip install -r requirements/dev.txt              # tests + lint (no torch)
  python -m venv .venv-eval
  .\.venv-eval\Scripts\python -m pip install -r requirements/dev.txt -r requirements/eval.txt   # + TextBlob, VADER
  python -m venv C:\sv
  C:\sv\Scripts\python -m pip install -r requirements/serve.txt -r requirements/dev.txt         # + torch (large)
  ```
  `.venv*` is already gitignored. `requirements/train.txt` (TensorFlow) must NOT be installed in the same
  environment as `serve.txt`.
- Run tests: `.\.venv-dev\Scripts\python -m pytest -q -p no:warnings`
  Run one file: `... pytest -q -p no:warnings tests/test_evaluation.py`
  Lint (this is exactly what CI blocks on): `... -m flake8 src tests app.py --count --select=E9,F63,F7,F82 --show-source --statistics`
- Run the modules with `PYTHONPATH=src` (PowerShell: `$env:PYTHONPATH="src"`), for example
  `python -m tweet_sentiment_classifier.evaluation.run --help`.
- **Shell quoting trap:** heredocs and multi-line strings that contain backslashes (`\n`, `\.`, `\`
  line-continuations) get corrupted by the tooling. Edit files with the editor/Write tool, not with `sed`,
  `python - <<EOF` or `echo`. For regexes in code, put them in a file via the editor tool.
- **Line endings:** many files are CRLF. A multi-line exact-match edit can fail only because of `\r\n`. Use the
  editor tool, which handles it; if you script an edit, read with `newline=""` and keep the original endings.
- Background commands that print "completed" may still have failed: always read the log's last lines and the
  exit code before trusting them.
- Large installs and training runs: run in the background, write a log file, check the log. Do not poll in a
  tight loop.
- Never print or commit secrets. `.env` is gitignored; `.env.docker` is tracked but only holds empty
  placeholders. Never paste a key into a file under version control.

---------------------------------------------------------------------------------------------------------

## 2. What this project is (one paragraph)

Three-class tweet sentiment (-1 negative, 0 neutral, 1 positive) with a training pipeline, a FastAPI app and a
research goal: **measure how much training on lexicon-generated labels inflates reported accuracy compared with
human judgement, and how cheaply a small human-labelled set repairs it.** The dataset
(`sentiment_analysis/dataset/thoroughly_cleaned_data.csv`, 162,969 tweets, ~91% mention "modi") was labelled by
TextBlob, so all current scores measure agreement with TextBlob ("silver" labels). A human-labelled "gold" set
is the missing evidence.

Repository map (only what you need):

| Path | What it is |
|---|---|
| `src/tweet_sentiment_classifier/components/` | pipeline stages: ingestion, validation, transformation (`clean_and_split`), trainer, evaluation (OLD), best-model selector (OLD), agents (LangChain search/report) |
| `src/tweet_sentiment_classifier/evaluation/` | `metrics.py` (all metrics, CIs, ECE, paired tests, `save_run`), `run.py` (one-command evaluation), `tune.py` (validation-only search), `seeds.py` (multi-split study) |
| `src/tweet_sentiment_classifier/models/` | `factory.py` (builds classifiers), `lexicon.py` (Majority, TextBlob, VADER) |
| `src/tweet_sentiment_classifier/pipeline/prediction_pipeline.py` | serving: `INFERENCE_MODE` local / llm / hybrid, reports `decision_source` |
| `src/tweet_sentiment_classifier/utils/llm_safety.py` | prompt-injection helpers, JSON validation |
| `app.py`, `templates/`, `static/` | FastAPI app and pages |
| `config/config.yaml`, `params.yaml`, `schema.yaml` | pipeline paths, hyperparameters, schema |
| `artifacts/evaluation/` | **authoritative results** (`results.csv/json`, soon `comparisons.csv`, `tuning.json`, `seed_study.*`) |
| `artifacts/data_transformation/split_manifest.json` | tracked description + fingerprints of the generated splits |
| `artifacts/model_evaluation/`, `artifacts/model_trainer/*.pkl` | OLD results and OLD models (trained on the leaky split) |
| `requirements/{base,api,serve,train,eval,dev}.txt` | pinned dependency groups |
| `tests/` | pytest suite; `conftest.py` has shared fakes |

---------------------------------------------------------------------------------------------------------

## 3. Known facts (verified earlier; re-derive only to check, and stop if they differ)

**Data / split** (from `artifacts/data_transformation/split_manifest.json`, seed 42):
raw 162,968 rows after validation; cleaned+deduplicated 159,583 (3 empty, 14 conflicting-label, 3,368 duplicate
rows removed). train / val / test = 111,707 / 15,959 / 31,917, stratified. Exact text overlap between splits = 0;
34 test texts are near-duplicates of train once digits are ignored. Test fingerprint `f1df771fdbf437c6`.
Label shares: 44% positive, 34% neutral, 22% negative (Majority-class accuracy 0.4448).

**Labels are TextBlob's**: `sign(polarity)` agrees with the dataset labels on 99.98% of 20,000 sampled rows;
TextBlob scores 0.9997 on the silver test set. VADER agrees only 57% (kappa 0.34).

**Baseline on the clean split, default hyperparameters** (single run, `artifacts/evaluation/results.csv`):

| Model | macro-F1 | 95% CI |
|---|---|---|
| TextBlob | 0.9997 | [0.9994, 0.9998] |
| LinearSVC (calibrated) | 0.9347 | [0.9319, 0.9372] |
| LogisticRegression (max_iter=100, does NOT converge) | 0.9032 | [0.8997, 0.9063] |
| RidgeClassifier (calibrated) | 0.9014 | [0.8981, 0.9046] |
| MultinomialNB | 0.7412 | [0.7360, 0.7460] |
| VADER | 0.5518 | [0.5463, 0.5568] |
| Majority | 0.2053 | [0.2035, 0.2069] |

**First tuning run (finished; `artifacts/evaluation/tuning.json`, uncommitted; validation macro-F1, test split
not read):**

| Model | Grid tried | Winner | Val macro-F1 | At grid edge? |
|---|---|---|---|---|
| LogisticRegression | C in {1, 4, 16, 64}, max_iter 1000 (all converged) | C=4 | 0.9244 (C=16: 0.9235) | no |
| LinearSVC | C in {0.1, 0.3, 1.0} | C=1.0 | 0.9337 | **yes, the largest tried** |
| MultinomialNB | alpha in {1, 0.3, 0.1, 0.03} | alpha=0.03 | 0.7377 | **yes, the smallest tried** |
| RidgeClassifier | alpha in {3, 1, 0.3} | alpha=1.0 | 0.8982 | no |

Two winners are on the edge of their grid, so a better value may lie outside it (WP1 step 3 extends the grids).
Untuned LogisticRegression (C=1) scored 0.9016 on validation, so tuning gained about 2.3 points for LR.

**Tests:** last committed state (`bb645d7`): 154 passed, 2 skipped (PyTorch test needs torch; real-lexicon test
needs TextBlob/VADER).

**Branch chain (all local, nothing pushed):**
`fix/honest-decider` > `fix/svc-confidence` > `fix/three-class-guard` > `chore/tests-ci` >
`fix/fresh-clone-pipeline` > `fix/real-monitoring` > `chore/hygiene-pinning` > `fix/async-limits` >
`feat/eval-baselines` > `feat/leakfree-split` > **`feat/tuning-significance` (current, has UNCOMMITTED work)**.

**Uncommitted work on `feat/tuning-significance`** (written, partly tested, never committed):
`metrics.py` (+`paired_bootstrap`, `mcnemar_exact`, `holm_adjust`, `save_run(..., comparisons)`),
`run.py` (+`compare_models`, writes `comparisons.csv`), `data_transformation.py` (now exposes pure
`clean_and_split`), new `evaluation/tune.py`, new `evaluation/seeds.py`, new
`tests/test_significance_and_tuning.py`. **One test fails** (see WP1 step 1).

---------------------------------------------------------------------------------------------------------

## 4. Work packages (do in this order)

Each WP lists: goal, steps, commands, **Acceptance** (all must be true), pitfalls. Suggested branch name in
brackets; create it from the previous WP's branch.

### WP1: Finish tuning, significance tests, seed study  [stay on `feat/tuning-significance`]

Goal: commit the in-progress work, produce the tuned hyperparameters and the final silver baseline.

1. **Fix the one failing test** `test_tune_picks_the_best_validation_setting_and_records_every_trial` in
   `tests/test_significance_and_tuning.py`. Cause (already diagnosed): the toy data in `toy()` is perfectly
   separable, so `alpha=1000` and `alpha=0.1` both reach macro-F1 1.0; ties keep the first listed setting, so
   `alpha=1000` "wins". The code is right; the test data is too easy. Fix `toy()` so classes overlap: give every
   document 2 words from its own class vocabulary and 3 from a pool shared by all classes. Before asserting,
   print the two validation scores and confirm alpha=1000 is *strictly* worse. Keep the assertion
   (`best == {"alpha": 0.1}`) and the LogisticRegression assertion (C=10 beats C=0.0001); if either fails,
   print the trials and adjust the *data*, not the assertion.
2. In `evaluation/run.py` function `_format`, after the model table, print the pairwise comparisons from
   `summary["comparisons"]`, one line each: `A vs B: macro-F1 diff +0.0xx [lo, hi]  Holm p=...`. Add a test with
   `capsys` that the line appears.
3. **Extend the two edge-of-grid searches, then re-run tuning** (about 5 minutes; background). In
   `evaluation/tune.py` `GRIDS`, change LinearSVC to `C in (0.3, 1.0, 3.0, 10.0)` and MultinomialNB to
   `alpha in (0.3, 0.1, 0.03, 0.01)`; keep the LR and Ridge grids. If a new winner is again at an edge, extend
   once more (at most twice) and say so in the commit. This uses the validation split only.
   ```powershell
   $env:PYTHONPATH="src"
   .\.venv-eval\Scripts\python -W ignore -u -m tweet_sentiment_classifier.evaluation.tune `
     --train artifacts/data_transformation/train.csv --val artifacts/data_transformation/val.csv `
     --out artifacts/evaluation/tuning.json
   ```
   `artifacts/evaluation/tuning.json` from the first run already exists in the working tree; the re-run
   overwrites it. If `train.csv`/`val.csv` do not exist (they are gitignored), regenerate them: run ingestion,
   validation and `DataTransformation` as in `tests/test_pipeline_smoke.py` (or ask the human for the files). The
   regenerated files must reproduce the fingerprints in `artifacts/data_transformation/split_manifest.json`.
4. **Apply the winners to `params.yaml`.** For each model set the best `params` from the new `tuning.json`
   (LogisticRegression: C and `max_iter: 1000`; LinearSVC: C; MultinomialNB: alpha; RidgeClassifier: alpha).
   Check the chosen LogisticRegression trial has `"converged": true`. Add a comment line:
   `# tuned on the validation split; see artifacts/evaluation/tuning.json`. Do not change anything else.
5. **Re-run the final evaluation** (this replaces the committed default-parameter table):
   ```powershell
   .\.venv-eval\Scripts\python -W ignore -m tweet_sentiment_classifier.evaluation.run `
     --train artifacts/data_transformation/train.csv --test artifacts/data_transformation/test.csv `
     --out artifacts/evaluation
   ```
   Confirm the printed `overlap_with_train=0.0%` and that `comparisons.csv` was written.
6. **Run the seed study** (about 6 to 10 minutes; background):
   `python -m tweet_sentiment_classifier.evaluation.seeds --data sentiment_analysis/dataset/thoroughly_cleaned_data.csv --seeds 42 1 2 3 4`
   **Integrity check:** the study's seed-42 test fingerprint must equal `f1df771fdbf437c6`
   (`artifacts/evaluation/seed_study.json` -> `per_seed` -> `42`). If it does not, STOP: the `clean_and_split`
   refactor changed behaviour. Report it.
7. Mutation-check the new code (break each, expect a failing test, restore): remove Holm adjustment; make
   `paired_bootstrap` use independent resamples for A and B; make `tune` pick the *lowest* score; make
   `save_run` keep a stale `comparisons.csv`.
8. Run the full suite and lint. Commit. Message: `feat: tune on validation, paired significance tests, seed study`.

**Acceptance:** all tests pass (expect about 175 passed, 2 skipped); `tuning.json`, `seed_study.csv/json`,
`comparisons.csv` and a new `results.csv` exist and are committed; `params.yaml` updated; LR converged; seed-42
fingerprint check passed; commit message states the *measured* validation winners and test macro-F1s (copy from
the files, do not paraphrase from memory).

### WP2: Replace the old evaluation path and retrain the deployed models  [`feat/unify-evaluation`]

Goal: nothing in the repo still reads or writes the old append-only `all_models_summary.csv`; the app serves
models trained on the clean split with tuned parameters.

1. `BestModelSelector` (`components/best_model_selector.py`): read `artifacts/evaluation/results.csv`.
   **Exclude `Majority`, `TextBlob`, `VADER`** (otherwise TextBlob, the label source, is "best"). Rank by
   `macro_f1`, tie-break lower `latency_p50_ms`. Keep the output keys (`best_model_name`, `metrics`,
   `tier_1_fast_fallback`) because `PredictionPipeline` reads `tier_1_fast_fallback`. Test with a temp CSV.
2. `ModelEvaluation` / `main.py` / MLflow: stop writing `all_models_summary.csv`. In `main.py`, after all
   models are trained, call `evaluation.run.run(...)` once (so one run id), then log its metrics to MLflow in
   the training environment. Keep DistilBERT handling untouched here (WP4a replaces it).
3. `/api/experiments` in `app.py`: read `artifacts/evaluation/results.csv` (columns listed in
   `metrics.save_run`) and expose `macro_f1`, `ci_low`, `ci_high`, `ece`, `latency_p50_ms`. Update
   `templates/pages/experiments.html` columns accordingly and update the existing tests in
   `tests/test_api.py` that write the old CSV format.
4. Add `src/tweet_sentiment_classifier/models/export.py`: `python -m ...models.export --models LogisticRegression LinearSVC MultinomialNB RidgeClassifier`
   fits each model with `models.factory.build_classifier` and `params.yaml` on `train.csv`, writes
   `artifacts/model_trainer/model_<Name>.pkl`, writes ONE `artifacts/data_transformation/preprocessor.pkl`
   (identical for all models because the TF-IDF settings are shared; assert that), and writes
   `artifacts/best_model/best_model_metadata.json` from the selector. Test on a tiny CSV in `tmp_path` and
   `monkeypatch.chdir(tmp_path)` (the paths are relative; do not overwrite the real artifacts in tests).
5. Run the export for real, then smoke-test serving: load `PredictionPipeline`, predict one sentence with
   `INFERENCE_MODE=local`, confirm `decision_source` is `local:<best model>` and confidence is a number.
6. **Ask the human before deleting** the stale tracked files in `artifacts/model_evaluation/`. Default: leave
   them and add a one-line `artifacts/model_evaluation/README.md` saying they are superseded.

**Acceptance:** `grep -r all_models_summary src app.py main.py templates tests` finds nothing; selector never
returns a lexicon model (test); experiments page numbers come from `results.csv`; the app answers `/predict`
with a model trained on the clean split; tests pass; lint clean.

### WP3: Gold set tooling (code and documents only; humans do the labelling)  [`feat/gold-tooling`]

Goal: everything needed so that two or three people can label tweets and the result is a frozen, hashed gold
set. **You do not label anything.**

Create `src/tweet_sentiment_classifier/gold/` with these modules, each with tests:

1. `docs/annotation_guidelines.md` (about 2 pages, mark it `DRAFT: needs human review`). Rules: label the
   sentiment the *author expresses*, not the political stance a reader infers (a neutral headline about an
   election result is neutral even if it favours a party); sarcasm: label the intended sentiment, set flag
   `sarcasm`; questions and pure news statements are neutral unless emotion is expressed; code-mixed/Hindi
   transliteration: label what you understand, set `code_mixed`, use `unclear` only if you cannot tell. Give at
   least 6 worked examples per class including hard cases. Tell annotators **not** to use any tool's score.
2. `gold/sample.py`: draw tweets **from the test split only**, stratified by silver label with equal counts per
   class (default 500 each = 1500; a `--pilot 100` option draws 100 balanced). Write:
   - `gold/to_annotate.csv` with columns `id,text` only (silver label hidden to avoid anchoring);
   - `gold/sample_private.csv` with `id,row_index,silver_label,sampling_weight` where
     `sampling_weight = natural_class_share / sampled_class_share` (needed later to estimate performance on the
     natural class distribution, because the sample is class-balanced);
   - `gold/sample_manifest.json` with seed, source fingerprint, counts.
3. `gold/label_studio_config.xml` (single-choice labels negative/neutral/positive + checkboxes for the four
   flags) and a CSV template for annotators who prefer a spreadsheet.
4. `gold/agreement.py`: reads one CSV per annotator (`id,label,flags`). Implement from scratch (no new
   dependency): pairwise Cohen's kappa, Krippendorff's alpha for nominal data (coincidence-matrix method),
   percentage agreement, a confusion matrix between annotator pairs, and flag rates. Tests: perfect agreement
   gives alpha 1.0; two raters who always disagree on a balanced binary task give a negative alpha; for two
   raters with many items alpha is within 0.01 of kappa; missing labels (an annotator skipped an item) are
   handled. Print a clear warning if alpha < 0.6 ("revise guidelines, re-pilot once").
5. `gold/adjudicate.py`: two annotators per item; where they agree the label stands; where they differ a third
   annotator's CSV breaks the tie; if still no majority, mark `unresolved` and exclude. Output `gold/gold.csv`
   with `id,text,sentiment,n_annotators,adjudicated`. Never overwrite the annotators' raw CSVs.
6. `gold/freeze.py`: join with `sample_private.csv`, assert no gold text appears in `train.csv` or `val.csv`,
   split into `gold_test.csv` (about 1000) and `gold_train_pool.csv` (about 500), stratified by gold label,
   with a fixed seed, and write `gold/gold_manifest.json` with SHA-256 fingerprints (use
   `evaluation.metrics.fingerprint`). Print: "gold_test must not be used for tuning."
7. Extend `evaluation.metrics.evaluate` with an optional `sample_weight` argument (weighted confusion matrix)
   and make `evaluation/run.py` accept `--weights-column`. Test against `sklearn` with `sample_weight`.

**Acceptance:** tests pass; running the tools end to end on a fake 60-row dataset in a temp directory produces
all files; the freeze step refuses to run if any gold text is in train/val (test it); the guidelines file
exists and is marked DRAFT. **Hand off to the human** with the exact instructions: run `sample.py --pilot 100`,
have 2 to 3 people label it independently, run `agreement.py`, report alpha.

### WP4: Experiments  [`feat/experiments`, one sub-branch per item if you prefer]

Prerequisite for everything below except 4a-smoke: a frozen gold set from WP3 (human work). Do **not**
fabricate a stand-in. Items that need a GPU or a key are marked; if you cannot run them, finish the code and
tests and write "NOT RUN" in the report.

Create `requirements/exp.txt` (`-r base.txt`, `-r eval.txt`, plus pinned `cleanlab`, `matplotlib`, `datasets`,
`groq` or the LangChain Groq package already pinned in `serve.txt`). Resolve versions with
`pip install --dry-run --report` and then install them to confirm they import; do not guess versions.

- **4a. PyTorch fine-tuning** `models/finetune.py`. Args: `--model`, `--train`, `--val`, `--max-len 64`,
  `--epochs`, `--lr`, `--batch-size`, `--seed`, `--out`. Map sentiment to ids with `label = sentiment + 1`
  and save with `id2label = {0: negative, 1: neutral, 2: positive}` so `PredictionPipeline` loads it (it
  refuses anything that is not exactly 3 labels). Select the best epoch on the **validation** split. Subsample
  option `--train-fraction` for CPU smoke runs. Test with the tiny random DistilBERT used in
  `tests/test_transformer_inference.py` (skip if torch is missing). Candidate models, **verify each exists
  with `huggingface_hub.model_info(<id>)` before using it and stop if it does not**:
  `cardiffnlp/twitter-roberta-base-2019-90m` (2019 Twitter RoBERTa; matches the data period),
  `distilbert-base-uncased`, `google/muril-base-cased` (Indian-language aware). Full runs need a GPU (Colab T4):
  write `docs/colab_finetune.md` with the exact commands. 3 seeds per model.
- **4b. Zero-shot LLM** `evaluation/llm_eval.py`: runs the hardened prompt from `utils/llm_safety.py` on a CSV,
  temperature 0, writes every raw response to a JSONL cache (so reruns cost nothing), handles rate limits with
  backoff, outputs predictions, then scores them with `evaluation.metrics`. Needs `GROQ_API_KEY` (human).
  Unit-test with a fake client. Unparseable replies count as errors and are reported, never silently labelled.
- **4c. Confident learning** `experiments/cleanlab_run.py`: 5-fold out-of-fold probabilities from the tuned
  LogisticRegression on `train.csv`; `cleanlab.filter.find_label_issues`; variants (a) drop flagged rows,
  (b) relabel to the predicted class; retrain; score on **val (silver) and gold_test**. Report the fraction
  flagged and how many flagged rows are ones where VADER disagrees with TextBlob.
- **4d. Gold budget** `experiments/gold_budget.py`: for k in {0, 50, 100, 200, 400} gold examples drawn from
  `gold_train_pool` with 5 seeds, compare (i) tuned LR trained on silver only, (ii) LR trained on silver plus the
  k gold rows with gold weight w in {1, 10, 50} (choose w by cross-validation *inside the pool*, never on
  gold_test), (iii) LR on the k gold rows only, (iv) the 4a transformer further fine-tuned on the k rows.
  Output a table of gold_test macro-F1 with CIs and a learning-curve PNG.
- **4e. Behavioural tests** `experiments/behavioural_tests.py`: template-generated cases for negation,
  contrast ("but"), intensifiers, domain-polar words ("unemployment is rising"), emoji, entity swap (replace
  one politician name with another: prediction should not change), and sarcasm. **A human writes or approves
  the expected labels**; you generate the templates and a CSV for review. Output failure rate per model and
  capability.
- **4f. Error taxonomy**: sample 200 gold test errors of the best silver-trained model into
  `experiments/error_sample.csv` with empty `category` and `notes` columns for a human to fill. Then count.
- **4g. Out-of-domain**: evaluate on TweetEval sentiment (`cardiffnlp/tweet_eval`, config `sentiment`;
  **check the label mapping** in the dataset card before use) and, if accessible and licence allows, SentiMix
  (Hinglish). Needs network.
- **4h. Ablations** (3 seeds each, on silver val and gold_pool CV, never gold_test): n-gram range (1,1) vs
  (1,2) vs (1,3); preprocessing on/off; training labels from TextBlob vs relabelled by VADER; backbone.
- **4i. Report assembler** `experiments/report.py`: reads `results.json` files and writes **Table 1** (silver
  macro-F1 vs gold macro-F1 per model, with CIs, as CSV and LaTeX) and the figures. Numbers must be read from
  files, never typed.

**Acceptance (per item):** code + tests merged; a results file under `artifacts/experiments/<item>/` produced
by a command recorded in `docs/commands.md`; every table cell traceable to that file; items you could not run
are listed as NOT RUN with the missing prerequisite.

### WP5: Engineering leftovers  [`chore/engineering-leftovers`]

1. **Blocking DB calls inside async handlers** (`app.py`): move the SQLAlchemy work into
   `await fastapi.concurrency.run_in_threadpool(...)`. Test that `/health` stays responsive while a slow fake
   DB commit runs (pattern: `tests/test_limits_and_async.py`).
2. **Postgres verification.** With Docker running: start `postgres:15-alpine` on port 55432, set
   `DATABASE_URL=postgresql://postgres:<pw>@localhost:55432/tweet_sentiment`, run `alembic upgrade head`, then
   run the API tests against it via an env-var-gated fixture (`TEST_DATABASE_URL`; skip when unset). The old
   bug was `strftime` (SQLite-only) in analytics; confirm `/api/analytics` returns data on Postgres.
3. **Docker build check:** `docker compose build` with `POSTGRES_PASSWORD` set; confirm the image installs
   CPU-only torch (`pip list | findstr torch` inside the container shows `+cpu`). If the build fails, report
   the exact error; do not guess.
4. **Lockfiles (optional):** `pip-compile` per requirements group if the human wants exact reproducibility.
5. **Ask the human** before: pushing branches, opening PRs, untracking `.env.docker`.

**Acceptance:** tests pass on SQLite and (when available) Postgres; docker build result recorded honestly.

### WP6: Honest demo  [`feat/silver-vs-gold-demo`]

1. New page `templates/pages/silver_vs_gold.html` and route `/silver-vs-gold`, fed by a JSON endpoint that
   reads `artifacts/evaluation/*.json` and the gold results. It must show: silver vs gold macro-F1 per model
   with CIs, the gold-budget curve, and a table of examples where TextBlob and the human label disagree.
   **No numbers hardcoded in HTML or JS.** Add a test that fails if a template contains a literal percentage
   or the strings from the audit's list of false claims.
2. Fix the landing page `templates/pages/landing.html`: remove or correct "Sub-100ms latency via local
   DistilBERT with Gemma LLM Fallback", "Enterprise", "Real-time", and "Powered by Enterprise Tech Stack"
   unless backed by a measurement in `results.csv` (latency p50 is measured there, per text).
3. Inspect `static/index.html`, `static/script.js`, `static/style.css`, `static/js/main.js`, `static/css/`
   (an older UI). If nothing references them, list them for the human; do not delete.

**Acceptance:** with no API keys the demo works; every displayed number comes from an artifact file.

### WP7: Paper  [`docs/paper`]

Only after WP4 produced real results. Create `paper/`:
- `outline.md`: Title, Abstract, Intro, Related work, Data and gold annotation, Methods, Experiments,
  Results (silver vs gold; gold budget; mitigation), Error analysis, Limitations, Ethics, Reproducibility.
- `tables/` generated by `experiments/report.py` (never hand-typed), `figures/` as PNG + the script.
- `references.bib` containing **only** the entries below. Before citing, open each link and confirm title,
  authors, year. Do not add a reference you have not opened.
  - Barbieri et al. 2020, TweetEval, Findings of EMNLP: https://aclanthology.org/2020.findings-emnlp.148
  - Rosenthal, Farra, Nakov 2017, SemEval-2017 Task 4: doi 10.18653/v1/S17-2088
  - Loureiro et al. 2022, TimeLMs, ACL demo: doi 10.18653/v1/2022.acl-demo.25
  - Hutto and Gilbert 2014, VADER, ICWSM: doi 10.1609/icwsm.v8i1.14550
  - Ribeiro et al. 2020, CheckList, ACL: doi 10.18653/v1/2020.acl-main.442
  - Northcutt, Jiang, Chuang 2021, Confident Learning, JAIR 70: doi 10.1613/jair.1.12125
  - Ratner et al. 2017, Snorkel, PVLDB 11(3): doi 10.14778/3157794.3157797
  - Patwa et al. 2020, SemEval-2020 Task 9 (SentiMix): doi 10.18653/v1/2020.semeval-1.100
  - Biswas, Young, Griffith 2022, *A Comparison of Automatic Labelling Approaches for Sentiment Analysis*: arXiv 2211.02976
  - Zhang et al. 2024, *Sentiment Analysis in the Era of Large Language Models: A Reality Check*: doi 10.18653/v1/2024.findings-naacl.246
  - Go, Bhayani, Huang 2009, Twitter sentiment classification using distant supervision (Stanford CS224N report; no DOI)
- `claims_checklist.md`: for every sentence in the abstract that contains a number, the file and key it comes
  from. A claim without a source is deleted, not softened.
- Venue shortlist: look up the *current* call for papers on the official site for each candidate (workshops such
  as WASSA, ICON, FIRE, ACL/EMNLP student workshops). Record URL, scope, deadline, page limit, and the date you
  checked. Do not state a deadline you did not read on the page. Avoid pay-to-publish venues.
- State the limitations plainly: single topic (91% "modi"), 2019 Indian political tweets, annotator pool size,
  class-balanced gold sample, silver labels from one lexicon.

**Acceptance:** `claims_checklist.md` is complete; the paper contains no number that is not in an artifact.

---------------------------------------------------------------------------------------------------------

## 5. Standard procedure for every WP (checklist to copy)

```
[ ] git status is clean; git branch --show-current shows the expected parent
[ ] git checkout -b <new-branch>
[ ] read every file you will touch
[ ] write failing test -> implement -> test passes
[ ] mutate the code on purpose -> test fails -> restore
[ ] full suite: all pass, count did not drop       (.venv-dev; plus C:\sv if you touched torch code)
[ ] flake8 blocking rules: 0
[ ] numbers in commit message / docs copied from output files
[ ] git add <specific files>; git commit -m "<type>: <what and why>"
[ ] write the report (section 7)
```

Commit message types: `fix:`, `feat:`, `chore:`, `docs:`, `ci:`. Body says *why*, and states anything NOT RUN.

---------------------------------------------------------------------------------------------------------

## 6. Needs a human (do not attempt; ask)

| Item | Why |
|---|---|
| Dataset licence and tweet redistribution terms (Kaggle "Twitter and Reddit Sentimental analysis", `Twitter_Data.csv`; the page text could not be read automatically) | decides what can be released with the paper |
| Annotators (2, ideally 3) and about 25 person-hours; reviewing the guidelines | the gold labels must be human |
| Expected labels for the behavioural tests | must be human-approved |
| GPU access (Colab T4) for fine-tuning | not available locally |
| `GROQ_API_KEY` and `HF_TOKEN` | LLM baseline and model downloads |
| Docker Desktop running | Postgres and image verification |
| Pushing branches, opening PRs, deleting tracked files (`artifacts/model_evaluation/*`, legacy `static/*`), untracking `.env.docker` | outward-facing or destructive |
| Venue choice, authorship, final claims | judgement calls |

---------------------------------------------------------------------------------------------------------

## 7. Report template (send after every WP)

```
WP<n> <name>: <DONE | PARTIAL | BLOCKED>
Branch / commit: <name> <short sha>
Changed: <files and one line each>
Tests: <N passed, M skipped, K failed> in <env>.  Mutations tried: <list> -> all caught? <yes/no>
Commands run + key outputs: <numbers copied from files>
NOT RUN / unverified: <list with the reason>
Decisions I made that you may want to reverse: <list>
Next WP and what it needs from you: <list>
```

---------------------------------------------------------------------------------------------------------

## 8. Definition of done for the whole project

- Phase 1 stabilised and Phase 2 baseline are committed and reproducible with the commands in README.
- A frozen, hashed, human-labelled gold set exists, with Krippendorff's alpha reported.
- Silver-vs-gold results with confidence intervals and significance tests exist for every model in the plan.
- The gold-budget curve and at least one mitigation (confident learning or gold fine-tuning) are evaluated on
  `gold_test`.
- The demo runs without keys and shows no hardcoded numbers.
- Every claim in the paper traces to a file; limitations are stated; items NOT RUN are listed, not hidden.
