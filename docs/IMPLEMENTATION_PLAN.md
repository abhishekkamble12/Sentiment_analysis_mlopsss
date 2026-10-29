# Implementation plan: from classifier to topic-monitoring platform with GenAI and an agent

Status: Phase 0 done, Phase 1 mostly done (analytics/model routers still query the DB; that moves into
`AnalyticsService` in Phase 4), Phase 2 done, React UI started early (it is now the product UI). 2026-10-29. Supersedes nothing: `docs/REMAINING_PLAN.md` (gold set, GPU fine-tuning, paper)
continues as the parallel research track.

## 1. What we are building

A **topic / public-opinion monitor**. A user registers a topic (a policy, brand, person or event). The system
keeps collecting public posts and news about it, classifies each one (fast local model, LLM for uncertain
posts), explains each prediction, groups posts into themes, shows trends, raises alerts on sudden shifts, and
offers an **agent** that researches a topic on request and answers questions about the collected data with
citations to real posts.

Ideas taken from the reference projects:

| From | Idea | Where it lands |
|---|---|---|
| kvj-085/sentiment_dashboard | continuous ingestion, trends over time | Phases 3, 4 (scheduler instead of Kafka) |
| GenAI-Customer-Review-Analyzer | structured LLM insight (theme, reason, suggestion), provider abstraction, layered backend, rate limit, cache, metrics | Phases 1, 5, 7 |
| pasted architecture | `/api/v1` routers, services layer, model loader at startup, highlighting, drift checks | Phases 1, 2, 4 |
| this repo | hybrid local/LLM inference, `llm_safety`, Alembic, evaluation rigour, gold-set tooling | kept and extended everywhere |

### Decisions taken by default (change them here before starting)

1. **Use case:** topic monitor (above). The local model was trained on TextBlob-labelled, mostly
   Modi-related tweets, so outside that domain its accuracy is unknown. The plan therefore measures it live
   (local-vs-LLM agreement, Phase 4) instead of assuming it.
2. **Frontend:** the FastAPI + Jinja pages stay the product UI; Streamlit stays the free hosted demo and calls
   the same services. React is optional (Phase 8) and only after Phase 6.
3. **LLM provider:** Groq (free tier, already integrated) by default, behind an interface so Claude, Gemini or
   OpenAI can be switched on with an environment variable.
4. **No Kafka.** A separate worker process with a scheduler is enough at this scale. Kafka is documented as the
   scale-out path, not built.
5. **Storage:** SQLite locally, Postgres in Docker/production (both already supported). Embeddings are stored
   in a table and searched with NumPy; pgvector is an optional upgrade, not a requirement.

## 2. Ground rules for every phase

- **One phase = one branch** (`feat/p1-services`, ...) merged to `main` only when its acceptance list is ticked
  and `python -m pytest -q` passes. The test count must never go down.
- **CI stays light.** CI installs `requirements/dev.txt` only (no torch, LangChain, network, API keys). Every
  new module that needs a heavy dependency imports it lazily, and every LLM, HTTP or embedding call goes
  through an interface that tests replace with a fake (extend `tests/conftest.py`, which already has
  `FakePipeline` and stub models). No test may call a real API.
- **Schema changes only through Alembic** (`alembic/versions/0003_...`), with a migration test like
  `tests/test_migrations.py`.
- **Untrusted text stays untrusted.** Every scraped post or user text that reaches a prompt goes through
  `sanitize_untrusted` / `wrap_untrusted`; every LLM JSON answer is validated before use
  (`extract_json_object` plus a schema). The LLM never writes SQL and never chooses URLs to fetch outside an
  allowlist.
- **Never invent results** (rule 1 of `REMAINING_PLAN.md`). The UI shows silver results labelled as silver;
  LLM labels are never stored as gold labels.
- **Respect the sources' terms.** Use official APIs or RSS, store only what is needed (text, URL, time, source),
  keep a retention setting (`POST_RETENTION_DAYS`, default 90).

## 3. Target architecture

```
                 Jinja pages (product UI)      Streamlit demo        (optional) React
                            \                       |                     /
                             v                      v                    v
   FastAPI  app/main.py   middleware: request id, logging, rate limit, CORS, /metrics
                            |
                    app/api/v1/*  routers  (predict, batch, topics, posts, analytics,
                            |                insights, alerts, agent, corrections, health)
                            v
                    app/services/*
   PredictionService   IngestionService   AnalyticsService   InsightService   AgentService
   (classify, explain,  (sources, dedupe,  (trends, drift,    (LLM extraction,  (LangGraph agent,
    abstain, log)        store, classify)   alerts)            clusters, digest) tools, traces)
          |                    |                  |                  |                 |
          v                    v                  v                  v                 v
   app/core/model_loader  app/sources/*     SQL queries       app/core/llm/*     app/agent/tools.py
   (pickles, once)        (gnews, reddit,                     (provider ABC,    (calls the services
                           youtube, rss)                       groq, claude...)  above, never raw SQL)
                                \_______________   ________________/
                                                v v
                              SQLAlchemy + Alembic  ->  SQLite (local) / Postgres (Docker)

   worker/  (separate process)  APScheduler: ingest every N min, insights hourly, drift + alerts, retention
```

The existing package `src/tweet_sentiment_classifier/` keeps the ML code (training, evaluation, gold,
experiments, `PredictionPipeline`, `llm_safety`). The new web layer lives in a new package
`src/tweet_sentiment_classifier/app/` and `app.py` at the root becomes a three-line shim
(`from tweet_sentiment_classifier.app.main import app`) so `uvicorn app:app`, the Dockerfile and the tests keep
working.

### New database tables (added across phases)

| Table | Phase | Key columns |
|---|---|---|
| `label_corrections` | 2 | text, predicted, corrected_by_user, decision_source, created_at, reviewed (bool) |
| `topics` | 3 | id, name, query, sources (JSON), language, active, created_at |
| `posts` | 3 | id, topic_id, source, external_id, url, text, author_hash, published_at, fetched_at, prediction, confidence, decision_source, content_hash; unique (source, external_id) and (topic_id, content_hash) |
| `ingestion_runs` | 3 | topic_id, source, started_at, finished_at, fetched, new, errors |
| `drift_snapshots` | 4 | topic_id, window_start, oov_rate, mean_confidence, class_shares (JSON), psi, llm_disagreement |
| `alerts` | 4 | topic_id, kind, window, baseline, observed, p_value, message, acknowledged |
| `post_embeddings` | 5 | post_id, model, vector (bytes) |
| `insights` | 5 | topic_id, window, cluster_id, label, sentiment, share, summary, example_post_ids (JSON), provider, prompt_version |
| `llm_cache` | 5 | key (hash of provider+model+prompt_version+input), response (JSON), created_at |
| `agent_runs` | 6 | id, kind, question, status, steps (JSON trace), tokens_in, tokens_out, cost_estimate, latency_ms |
| `annotation_queue` | 6 | post_id, reason (uncertain / disagreement), priority, status |

## 4. Phases

Estimates assume one person part-time-heavy (about 5 h/day). Total about 7 to 8 weeks to Phase 7.

---------------------------------------------------------------------------------------------------------

### Phase 0: fix the Streamlit demo (1 to 2 days) — branch `fix/streamlit-demo`

Findings from the analysis of `streamlit_app.py`:

1. Streamlit Cloud installs the root `requirements.txt` = `serve.txt` (torch, transformers, LangChain). Add
   `requirements/streamlit.txt` (`-r base.txt` + pinned `streamlit`) and point Cloud at it. Streamlit Cloud reads
   the dependency file next to the entrypoint, so move the app to `streamlit/streamlit_app.py` with
   `streamlit/requirements.txt` containing `-r ../requirements/streamlit.txt`. Keep artifact paths resolved
   from the repo root (`Path(__file__).resolve().parents[1]`) so the working directory does not matter.
   Verify this behaviour against the current Streamlit Cloud docs before relying on it.
2. Replace `use_container_width=True` (4 places) with `width="stretch"`; pin `streamlit==<installed version>`
   in one place and remove `streamlit>=1.35.0` from `requirements/api.txt` (the API image does not need it).
3. Use `clean_text` from `components/data_transformation.py` instead of `.lower().strip()`.
4. Add the silver-label caveat to the header and the benchmarks tab; mark TextBlob as "label source, not a
   competitor" in the chart (or drop it from the chart and keep it in the table).
5. Keep batch results in `st.session_state` so they survive the download click; enforce `MAX_CSV_ROWS`.
6. Read the calibration wrapper's type from the loaded model instead of hard-coding "CalibratedClassifierCV".
7. Fix `docs/streamlit_deployment.md` ("progress bar", new paths and requirements file).

**Tests:** `tests/test_streamlit_app.py` with `streamlit.testing.v1.AppTest`: loads with no exception; sample
button fills the text box; Analyze shows a label; a CSV with 3 rows produces 3 predictions. Skip the module
when `streamlit` is not installed (`pytest.importorskip`).
**Acceptance:** app runs locally with only `requirements/streamlit.txt` installed in a fresh venv; deployed URL
works; tests pass.

---------------------------------------------------------------------------------------------------------

### Phase 1: services layer and `/api/v1` (4 to 5 days) — branch `feat/p1-services`

Goal: same behaviour, new structure. No feature changes in this phase.

```
src/tweet_sentiment_classifier/app/
  main.py               create_app(): middleware, routers, lifespan (load models once)
  deps.py               get_db, get_prediction_service, ... (FastAPI Depends; tests override these)
  core/model_loader.py  loads preprocessor + best model + metadata once; exposes .vectorizer, .model, .meta
  core/settings.py      re-export of config/env_config.Settings plus new keys
  services/prediction_service.py   wraps PredictionPipeline: validate, cache, predict, log
  services/batch_service.py        current _process_csv logic
  services/analytics_service.py    current /api/analytics, /monitoring, history queries
  services/report_service.py       current search-analysis flow (SearchAgent + ReportAgent)
  api/v1/{predict,batch,analytics,reports,results,feedback,health}.py
  api/pages.py          the Jinja page routes
```

Tasks:
1. Move code out of `app.py` route by route, starting with `/health`, then predict, CSV, analytics, search
   analysis, results, feedback. After each move run the full suite.
2. Keep every existing URL working: mount the new routers under `/api/v1/...` **and** keep the old paths as thin
   aliases (`/predict`, `/api/predict`, `/upload_csv`, ...) marked deprecated in OpenAPI, so the Jinja JS and
   the tests do not break. Update `static/js/main.js` to the new paths in a separate commit.
3. Replace module-level globals (`pipeline`, `search_agent`, `predict_cache`) with objects created in the
   lifespan handler and stored on `app.state`, reached through `deps.py`.
4. Replace the per-process dict caches with a small `TTLCache` class in `core/cache.py` (bounded size, expiry),
   later swappable for Redis.
5. Add request-id middleware (`X-Request-ID`, logged on every line) and a consistent error body
   `{"error": {"code": ..., "message": ..., "request_id": ...}}`.
6. Streamlit calls `PredictionService` directly (no HTTP), so both UIs share one code path.

**Tests:** all existing tests pass unchanged (they are the safety net); add `tests/test_api_v1.py` covering
the new paths and the aliases returning identical JSON.
**Acceptance:** `app.py` is under 20 lines; no route function contains business logic; `uvicorn app:app` and
Docker still work.

---------------------------------------------------------------------------------------------------------

### Phase 2: explain, abstain, correct (3 days) — branch `feat/p2-explain`

1. **Explanations** (`services/explain.py`). The served model is `CalibratedClassifierCV(LinearSVC)`, sigmoid,
   3 folds; each fold's `estimator.coef_` has shape (3, 15000). Average the fold coefficients once at load time
   (`W`). For a text with TF-IDF vector `x`, the contribution of n-gram `j` to class `c` is `x[j] * W[c, j]`.
   Return the top-k positive and negative contributors for the predicted class plus their character spans in
   the original text for highlighting. Checked on 2026-10-29: for "worst policy ever but the speech was
   great", `worst` contributes +2.63 to negative and `great` +2.43 to positive. These are exact for the
   linear margin; calibration is monotone per class, so the ranking is what to show — say so in the UI.
   For models without `coef_`, return `explanation: null` (do not fake one). Add SHAP only if a non-linear
   model is ever served.
2. **Abstain.** New setting `ABSTAIN_THRESHOLD` (default: none). Below it the response has
   `"prediction": "uncertain"` in `local` mode, or goes to the LLM in `hybrid` mode (already implemented).
   Pick the threshold on the validation split from the existing calibration data, never on test.
3. **Corrections.** `POST /api/v1/corrections` stores `label_corrections`; a "Wrong label?" control on the
   predict page and in Streamlit. These rows are candidates for annotation, never training labels on their own.
4. Predict page: highlighted words (green/red) and the top contributors list.

**Tests:** contribution sums match `W @ x` for a stub linear model; spans map back to the original text with
punctuation and URLs; `explanation` is `null` for a model without `coef_`; abstain triggers below threshold;
correction endpoint validates the label set and length limits.
**Acceptance:** explanations add < 2 ms p50 (measure and record).

**Done (2026-10-29).** Measured on the served model: 0.84 ms per explanation (mean of 200 calls). The served
model averages three sigmoid-calibrated folds, so each fold's coefficients are scaled by its calibrator slope
before averaging and the result is labelled approximate. Abstain threshold chosen on validation
(`artifacts/evaluation/abstain.json`): 0.7273, coverage 86.9%, kept silver accuracy 97.0%; off by default.
Corrections are on the React predict page only (the Streamlit demo has no database); Streamlit shows the
term table. Rows logged as `uncertain` appear as neutral on the React history/dashboard pages until those
pages learn the label.

---------------------------------------------------------------------------------------------------------

### Phase 3: topics and continuous ingestion (5 days) — branch `feat/p3-ingestion`

1. **Source adapters** (`app/sources/`), one interface:
   `class Source: name: str; def fetch(self, query: str, since: datetime) -> list[RawPost]`.
   - `google_news.py`: Google News RSS search (`feedparser`, no key).
   - `rss.py`: any allowlisted feed, filtered with the existing `is_relevant`.
   - `reddit.py`: official API via PRAW (needs a free Reddit app id/secret; skipped if absent).
   - `youtube.py`: YouTube Data API v3 comments (needs `YOUTUBE_API_KEY`; quota-aware: stop at a configured
     daily budget).
   - X/Twitter is not included: its API is paid. Document how to add it as one more adapter.
   Every adapter: timeouts, retries with backoff, per-source rate limit, returns normalised `RawPost`
   (external_id, url, text, author, published_at). Hash the author (`author_hash`), do not store handles.
2. **IngestionService:** for each active topic and source: fetch since last run, drop duplicates
   (unique keys + near-duplicate key from `_near_duplicate_key`), classify new posts in batch through
   `PredictionService` (local model in batch; hybrid LLM calls capped per run by `MAX_LLM_CALLS_PER_RUN`),
   store, record an `ingestion_runs` row. One failing source never stops the others.
3. **Worker** (`src/tweet_sentiment_classifier/worker/main.py`): APScheduler `BlockingScheduler`; jobs:
   ingest every `INGEST_INTERVAL_MIN` (default 30), retention cleanup daily. A DB advisory lock (Postgres) or a
   lock row (SQLite) prevents two workers running the same job. Add a `worker` service to
   `docker-compose.yml` using the same image with `command: python -m tweet_sentiment_classifier.worker.main`.
   For hosts without background workers (free tiers), also expose `POST /api/v1/ingest/run` protected by
   `INGEST_TOKEN`, callable from a GitHub Actions cron.
4. **API + UI:** CRUD for topics (`/api/v1/topics`), `GET /api/v1/topics/{id}/posts` (filters: source,
   sentiment, date range; pagination), a "Topics" page with a live feed of the latest posts.

**Tests:** each adapter parses a recorded fixture response (stored under `tests/fixtures/sources/`, no
network); dedupe across runs; one adapter raising does not stop the run; LLM cap respected; migration test for
the new tables; topic CRUD validation.
**Acceptance:** with Google News only (no keys), registering a topic and running one ingestion stores
classified, de-duplicated posts; the worker runs for an hour locally without errors (check its log).

---------------------------------------------------------------------------------------------------------

### Phase 4: trends, drift and alerts (3 to 4 days) — branch `feat/p4-analytics`

1. **Trends** (`AnalyticsService`): per topic and time bucket (hour/day): volume, class shares with Wilson
   95% intervals, mean confidence, split by source. Endpoint `GET /api/v1/topics/{id}/trends?bucket=day`.
2. **Drift**, computed by a worker job per topic and window into `drift_snapshots`:
   - out-of-vocabulary rate (share of tokens not in the vectorizer vocabulary): the clearest sign the topic is
     outside the training domain;
   - mean confidence and share below the abstain threshold;
   - PSI of the class distribution against the topic's first week (or a chosen baseline);
   - **local-vs-LLM disagreement**: send a small random sample of posts (setting `AUDIT_SAMPLE_RATE`,
     default 0.05, capped per day) to the LLM and record the disagreement rate. This is the live quality signal;
     label it "agreement with the LLM", not accuracy.
3. **Alerts:** a two-proportion z-test of the negative share in the latest window against the baseline window,
   fired only when volume ≥ `ALERT_MIN_POSTS` and the adjusted p-value < 0.01, with a cooldown per topic.
   Also alert on volume spikes and on drift thresholds. Store in `alerts`; show on the dashboard; optional
   email/Telegram notifier behind an interface.
4. **Dashboard page** per topic: trend chart, class shares, volume, drift panel with plain explanations,
   alert list. Reuse `templates/pages/analytics_dashboard.html` and its chart component.

**Tests:** trend aggregation on a seeded fixture DB gives hand-computed numbers; Wilson interval values;
OOV rate on a stub vocabulary; alert fires on a constructed jump and does not fire under the minimum volume or
during cooldown.
**Acceptance:** dashboard shows a topic's trend and drift from real ingested data.

---------------------------------------------------------------------------------------------------------

### Phase 5: GenAI insights (5 days) — branch `feat/p5-insights`

1. **Provider layer** (`app/core/llm/`): `class LLMProvider(ABC): def complete_json(self, system, user,
   schema, max_tokens) -> dict`, with `groq.py` (wraps the existing client), `anthropic.py`, `gemini.py`,
   `openai.py`, and `fake.py` for tests. Selected by `LLM_PROVIDER` / `LLM_MODEL`. Every call: timeout, retry,
   token and cost accounting, `llm_cache` lookup keyed by provider + model + prompt version + input hash.
   Move the existing Hugging Face/Groq sentiment calls in `PredictionPipeline` onto this layer.
2. **Prompts as files** (`app/prompts/*.md`, each with a version string) so outputs are traceable and prompt
   changes show up in diffs.
3. **Structured extraction**, Pydantic schemas validated after `extract_json_object`; invalid output is retried
   once, then stored as a failure (never a made-up value):
   - per post (only on demand or for the audit sample): `aspects[{aspect, sentiment, evidence_span}]`,
     `reason`, `entities`;
   - per cluster (the main use, cheap): `label` (2 to 4 words), `sentiment`, `summary`, `representative_quotes`
     (must be exact substrings of member posts: verify, drop any that are not).
4. **Embeddings and themes:** `sentence-transformers` (`all-MiniLM-L6-v2`, add to `serve.txt`, imported
   lazily) into `post_embeddings`; hourly job clusters each topic's recent posts with `sklearn.cluster.HDBSCAN`
   (already in scikit-learn, no new dependency), then one LLM call per cluster to name and summarise it. Result:
   "fuel prices (negative, 34% of negative posts)".
5. **Daily digest:** one LLM call compares today's clusters, trends and alerts with yesterday's and writes a
   short "what changed" summary with links to posts. Stored in `insights`, shown on the topic page.
6. **Limits:** `LLM_DAILY_TOKEN_BUDGET` enforced in the provider layer; rate limit on the on-demand endpoints.

**Tests:** fake provider returns fixed JSON; invalid JSON path stores a failure; quotes that are not substrings
are dropped; cache hit avoids a second call; budget exhaustion stops calls and says so; clustering job on
fixture embeddings (no model download in CI).
**Acceptance:** for a real topic, themes and the daily digest appear on the dashboard; token use per day is
visible on the monitoring page.

---------------------------------------------------------------------------------------------------------

### Phase 6: the agent (7 to 8 days) — branch `feat/p6-agent`

Replace the fixed `SearchAgent` → `ReportAgent` chain with a tool-using agent built on **LangGraph**
(LangChain is already a dependency; pin the version you install in `serve.txt`). The current chain stays as the
fallback when no tool-calling model is configured.

1. **Tools** (`app/agent/tools.py`): thin, typed wrappers around the services. The agent never sees the
   database or the network directly.

   | Tool | Does |
   |---|---|
   | `list_topics()` | registered topics |
   | `search_posts(topic, query, sentiment?, since?, until?, k)` | semantic + keyword search over stored posts (embeddings from Phase 5); returns ids and text |
   | `get_trend(topic, bucket, since?, until?)` | numbers from AnalyticsService |
   | `compare_periods(topic, a, b)` | share changes with intervals |
   | `get_themes(topic, window)` | Phase 5 clusters |
   | `get_alerts(topic, since?)` | Phase 4 alerts |
   | `fetch_fresh(topic_or_query, sources)` | runs ingestion now for allowlisted sources, capped |
   | `classify(texts)` | PredictionService, batch, capped |

2. **Graph:** `plan → act (tool calls) → observe → (loop) → write answer → verify citations → done`.
   Guardrails: max 8 steps, max tool calls per tool, wall-clock timeout, token budget per run; tool outputs
   wrapped with `wrap_untrusted` because they contain scraped text; the final answer must cite post ids, and a
   verifier node drops or flags any claim whose cited id was not returned by a tool in this run. Every run is
   saved to `agent_runs` (full trace, tokens, cost, latency).
3. **Two entry points:**
   - **Research report** `POST /api/v1/agent/report {topic | query, period}`: runs in the background, returns a
     run id; `GET /api/v1/agent/runs/{id}` streams progress (SSE) and returns the report: summary, key themes,
     what changed, notable posts with links, limitations (domain shift, silver labels, sample size).
   - **Ask your data** `POST /api/v1/agent/ask {question, topic?}`: chat page; answers like "Why did
     negativity rise on Tuesday?" with numbers from tools and cited posts.
4. **Human in the loop / active learning:** a worker job fills `annotation_queue` with the most uncertain posts
   and the local-vs-LLM disagreements; an "Annotate" page exports them in the Label Studio format from
   `docs/label_studio_config.xml`. This connects the product to the gold-set work; labels still come from
   people (`REMAINING_PLAN.md` rule 3).
5. **Agent evaluation** (`tests/agent_eval/` + `python -m tweet_sentiment_classifier.agent.evaluate`):
   - about 25 questions over a fixed, seeded fixture database, each with the ground truth computed by SQL
     (numeric answers), the tools that should be used, and posts that must be cited;
   - metrics: answer correctness (numbers within tolerance), tool-selection accuracy, citation validity (share
     of cited ids that exist and support the claim), steps, tokens, cost, latency;
   - CI runs the harness with a scripted fake model (checks the plumbing and guardrails); the real-model run is
     manual with a key, and its output is written to `artifacts/agent_eval/` with the model id and date. Report
     only numbers from runs you actually made.

**Tests:** each tool validates input and caps sizes; step and budget limits stop a looping fake model;
injected instructions inside a post ("ignore previous instructions...") do not change tool calls (fake model
scripted to try); citation verifier removes fabricated ids; run trace persisted.
**Acceptance:** a report and an ask-your-data answer for a real topic, both with valid citations; the agent
eval table recorded for at least one real model.

---------------------------------------------------------------------------------------------------------

### Phase 7: production hardening and deployment (4 days) — branch `feat/p7-prod`

1. **Auth:** API keys (hashed in a table) for write endpoints, the agent and ingestion; read-only demo mode
   without a key. Rate limits per key/IP (`slowapi`, Redis-backed when `REDIS_URL` is set, in-memory otherwise;
   fail open if Redis is down).
2. **Observability:** Prometheus `/metrics` (request latency, predictions by decision source, LLM tokens and
   cost, ingestion counts and errors, agent runs); Grafana dashboard JSON in `monitoring/grafana/`; LLM/agent
   tracing with Langfuse (self-hosted or cloud) behind a setting, off by default.
3. **Docker Compose:** `web`, `worker`, `db` (Postgres, add the `pgvector` image only if you adopt it),
   optional `redis`, `prometheus`, `grafana`. Health checks for each. Run `alembic upgrade head` on start.
4. **CI:** add the migration check, the agent-eval harness with the fake model, a Docker build job, and
   `pip-audit` on the requirement files.
5. **Deploy:** backend + worker on Render/Railway/Fly (or the GitHub Actions cron fallback from Phase 3),
   Postgres on Neon, Streamlit demo on Streamlit Cloud. Document the env vars in `.env.example`.
6. **README rewrite:** what it does, architecture diagram, screenshots/GIF, live demo link, honest
   limitations section (silver labels, domain shift, LLM agreement is not accuracy), how to run, the evaluation
   and agent-eval tables.

**Acceptance:** `docker compose up` brings up the full stack from a clean clone; the public demo works; a
fresh clone following the README reaches a running app.

---------------------------------------------------------------------------------------------------------

### Phase 8 (optional): React frontend (2 to 3 weeks)

Only after Phase 6. Vite + React + TypeScript in `frontend/`, calling `/api/v1` (CORS for port 5173 is already
configured in `app.py`). Pages: topics, topic dashboard, post feed, predict with highlighting, agent chat with
streamed steps, alerts, annotate. Generate the API client from FastAPI's OpenAPI schema. Deploy on Vercel or
Netlify. Retire the Jinja pages only when every page has a React equivalent.

---------------------------------------------------------------------------------------------------------

### Parallel track: the gold set (from `REMAINING_PLAN.md`)

Run alongside Phases 1 to 7: annotation guidelines review, pilot, annotation, adjudication, freeze, gold
evaluation. When `artifacts/evaluation/gold/results.json` exists, the dashboard and README show accuracy
against human labels next to the silver numbers. Phase 6's annotation queue can feed later annotation rounds
(not the frozen gold test set).

## 5. Dependencies to add

| Package | File | Phase | Note |
|---|---|---|---|
| `streamlit` (pinned) | `requirements/streamlit.txt` | 0 | removed from `api.txt` |
| `APScheduler` | `api.txt` | 3 | worker |
| `praw` | `serve.txt` | 3 | optional Reddit adapter |
| `google-api-python-client` | `serve.txt` | 3 | optional YouTube adapter |
| `sentence-transformers` | `serve.txt` | 5 | lazy import, uses the torch already in `serve.txt` |
| `langchain-anthropic` / `google-genai` / `openai` | `serve.txt` | 5 | optional providers |
| `langgraph` | `serve.txt` | 6 | agent |
| `slowapi`, `redis`, `prometheus-client` | `api.txt` | 7 | `redis` optional at runtime |

Pin each to the version you install and test with, like the existing files. `dev.txt` stays free of all of
them; tests use fakes.

## 6. Risks and how the plan handles them

| Risk | Handling |
|---|---|
| Local model is weak outside its training domain | OOV rate and local-vs-LLM disagreement shown per topic (Phase 4); hybrid mode routes uncertain posts to the LLM; gold set and fine-tuning in the research track |
| LLM cost or free-tier limits | caps per run and per day, cache, cluster-level instead of post-level calls, budget shown on the monitoring page |
| Source APIs change or block | adapters isolated, recorded fixtures in tests, failures recorded per run and never fatal |
| Prompt injection through scraped posts | existing `llm_safety` wrapping, tool allowlist, no LLM-written SQL or URLs, injection tests in Phase 6 |
| Agent makes things up | citation verifier, answers built from tool outputs, agent eval with citation validity |
| Scope creep | each phase ships alone; React and Kafka are explicitly optional |

## 7. Milestones

| Week | Done when |
|---|---|
| 1 | Phase 0 deployed demo; Phase 1 merged (same behaviour, new structure) |
| 2 | Phase 2 (explanations, abstain, corrections) |
| 3 | Phase 3 (topics ingesting on a schedule) |
| 4 | Phase 4 (trends, drift, alerts dashboard) |
| 5 | Phase 5 (themes, digest, provider layer) |
| 6–7 | Phase 6 (agent, ask-your-data, annotation queue, agent eval) |
| 8 | Phase 7 (hardening, deployment, README) |
| later | Phase 8 React, if wanted |
