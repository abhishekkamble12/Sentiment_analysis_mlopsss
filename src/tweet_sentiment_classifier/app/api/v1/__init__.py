from fastapi import APIRouter

from tweet_sentiment_classifier.app.api.v1 import analytics, models, predict, topics

router = APIRouter()
for module in (predict, analytics, models, topics):
    router.include_router(module.router)
