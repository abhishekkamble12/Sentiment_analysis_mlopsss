# Deployment

What is verified and what is not: the application, its migrations (0001 to 0008 on an empty SQLite database), the
compose file's syntax and the CI workflow's syntax were checked. **The Docker image has not been built and the
stack has not been started on Docker or any host** (the Docker daemon was not running when this was written), and
nothing has been deployed. Treat the steps below as a checklist to verify, not as a tested recipe.

## Pieces

| Piece | Runs as | Needs |
|---|---|---|
| Web app + API + Jinja pages + React app (`/app`) | `web` container (`uvicorn`, one worker) | Postgres |
| Collection, themes, annotation queue | `worker` container (or the cron workflow below) | same image and database |
| Database | Postgres (compose `db`, or a hosted one such as Neon) | `DATABASE_URL` |
| Metrics | `prometheus` (compose profile `monitoring`) | `METRICS_TOKEN` if the port is reachable |
| Streamlit demo | Streamlit Cloud, main file `streamlit_demo/streamlit_app.py` | nothing else (it has its own requirements) |

## Settings that matter in production

Put these in the host's environment (never in a file under version control). `.env.docker` already turns on
`API_AUTH`, rate limits and the hardening defaults.

| Setting | Why |
|---|---|
| `DATABASE_URL` | Postgres URL. SQLite works for a single machine but not across `web` and `worker` containers. |
| `POSTGRES_PASSWORD` | Required by `docker-compose.yml`. |
| `INGEST_TOKEN` or API keys | Without one of them the ingestion, insight and agent endpoints answer 503 on purpose. |
| `API_AUTH=true` | Topic create/delete and the annotation refresh then also need a key. |
| `RATE_LIMIT_*` | Per client IP. Set `TRUST_PROXY=true` **only** behind a proxy you control that sets `X-Forwarded-For`; otherwise anyone can spoof it. |
| `METRICS_TOKEN` | `/metrics` is open without it. Also put the token in `monitoring/prometheus.yml`. |
| `GROQ_API_KEY` (or `LLM_PROVIDER` + key + `LLM_MODEL`) | Turns on themes naming, the daily note and the agent. |
| `LLM_DAILY_TOKEN_BUDGET`, `AGENT_MAX_TOKENS` | The ceiling on what the LLM features can spend per day and per session. |

Create a key for yourself or a cron job (the key is printed once; only its hash is stored):

```bash
docker compose exec web python -m tweet_sentiment_classifier.app.core.auth create "cron"
docker compose exec web python -m tweet_sentiment_classifier.app.core.auth list
docker compose exec web python -m tweet_sentiment_classifier.app.core.auth revoke 1
```

## Docker Compose

```bash
export POSTGRES_PASSWORD='choose-one'
docker compose up --build                      # db, web (migrates first), worker
docker compose --profile monitoring up -d      # adds Prometheus on 127.0.0.1:9090
```

The worker starts only after the web container reports healthy, because the web container runs `alembic upgrade head`.
Run exactly one worker: it has no lock.

## Hosts without background workers (free tiers)

Run only the web container and let GitHub call it: `.github/workflows/ingest.yml` posts to
`/api/v1/ingest/run` every 30 minutes. Set the repository variable `APP_URL` and the secret `INGEST_API_KEY`.
That job collects posts; themes and the daily note are built by the worker, so on this setup call
`POST /api/v1/topics/{id}/insights/run` (with the key) when you want them refreshed.

Free hosts sleep when idle: the first request after a pause can take 30 seconds or more.

## One process only

The rate limiter, the prediction cache and the in-flight agent sessions live in the web process. Running more than
one web replica gives each its own limits and caches, and an agent session started on one replica is polled on
whichever replica receives the request (the session itself is stored in the database, so polling still works, but
the limits multiply). Move the limiter counters to Redis before scaling out.

## Security checklist

- [ ] `INGEST_TOKEN` set or at least one API key created; `API_AUTH=true`.
- [ ] `TRUST_PROXY` matches reality; the app is served over HTTPS by the host or proxy.
- [ ] `/metrics` not reachable from the internet, or `METRICS_TOKEN` set.
- [ ] Database not published on a public port (compose binds it to `127.0.0.1`).
- [ ] LLM keys only in the host's environment; `LLM_DAILY_TOKEN_BUDGET` set to a number you are happy to pay for.
- [ ] `.env` and `.env.docker` contain no real secrets in version control (`.env.docker` is tracked: it holds placeholders only).

## Not done

Redis-backed limits, Grafana dashboards, LLM tracing (for example Langfuse), login accounts (there are shared
keys, not users), and any real deployment.
