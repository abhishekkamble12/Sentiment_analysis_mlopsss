"""Background worker: collect posts for every active topic on a schedule and purge old ones.

    python -m tweet_sentiment_classifier.worker.main            # loops forever
    python -m tweet_sentiment_classifier.worker.main --once     # one pass, then exit

shortcut: a plain sleep loop and no lock, so run exactly one worker; switch to APScheduler plus a DB lock when
Phase 4 adds more jobs or when more than one worker is needed.
"""
import argparse
import asyncio
import logging
import time

from tweet_sentiment_classifier.app.container import Container
from tweet_sentiment_classifier.app.services import ingestion_service, insight_service
from tweet_sentiment_classifier.config.database import Base, SessionLocal, engine
from tweet_sentiment_classifier.entity.models import Topic

logger = logging.getLogger("worker")


async def run_once(c) -> None:
    with SessionLocal() as db:
        results = await ingestion_service.ingest_all(c, db)
        for topic, runs in results.items():
            logger.info(f"topic '{topic}': " + ", ".join(
                f"{r['source']} fetched={r['fetched']} new={r['new']}" + (f" ERROR {r['error']}" if r["error"] else "")
                for r in runs))
        for topic in db.query(Topic).filter(Topic.active.is_(True)).all():
            try:
                done = insight_service.refresh_topic(db, topic, c.llm, c.settings)
                if done["themes"] != "skipped":
                    logger.info(f"topic '{topic.name}' insights: {done}")
            except Exception:  # one topic's insight failure must not stop the others or the purge
                db.rollback()
                logger.exception(f"insights failed for topic '{topic.name}'")
        purged = ingestion_service.purge_old_posts(db, c.settings.POST_RETENTION_DAYS)
        if purged:
            logger.info(f"purged {purged} posts older than {c.settings.POST_RETENTION_DAYS} days")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    Base.metadata.create_all(bind=engine)
    c = Container.build()
    while True:
        try:
            asyncio.run(run_once(c))
        except Exception:  # keep the worker alive; the next pass retries
            logger.exception("ingestion pass failed")
        if args.once:
            return
        time.sleep(c.settings.INGEST_INTERVAL_MIN * 60)


if __name__ == "__main__":
    main()
