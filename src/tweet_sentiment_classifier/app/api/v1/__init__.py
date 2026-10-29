from fastapi import APIRouter

from tweet_sentiment_classifier.app.api.v1 import analytics, models, predict

router = APIRouter()
for module in (predict, analytics, models):
    router.include_router(module.router)
