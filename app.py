"""ASGI entry point (`uvicorn app:app`). The application lives in tweet_sentiment_classifier.app."""
import uvicorn

from tweet_sentiment_classifier.app.main import create_app

app = create_app()
container = app.state.container

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, workers=1, log_level="info")
