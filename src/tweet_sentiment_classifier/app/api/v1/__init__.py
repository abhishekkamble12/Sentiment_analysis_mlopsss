from fastapi import APIRouter

from tweet_sentiment_classifier.app.api.v1 import analytics, insights, models, predict, topic_analytics, topics

router = APIRouter()
for module in (predict, analytics, models, topics, topic_analytics, insights):
    router.include_router(module.router)
