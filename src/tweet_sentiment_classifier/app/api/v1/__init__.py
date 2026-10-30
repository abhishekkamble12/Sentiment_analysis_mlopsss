from fastapi import APIRouter

from tweet_sentiment_classifier.app.api.v1 import analytics, models, predict, topic_analytics, topics

router = APIRouter()
for module in (predict, analytics, models, topics, topic_analytics):
    router.include_router(module.router)
