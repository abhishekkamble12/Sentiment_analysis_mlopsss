from fastapi import Request

from tweet_sentiment_classifier.config.database import get_db  # noqa: F401  (re-exported for routers)


def get_container(request: Request):
    return request.app.state.container
