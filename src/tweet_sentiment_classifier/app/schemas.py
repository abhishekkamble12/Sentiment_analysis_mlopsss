from typing import Literal

from pydantic import BaseModel, Field

from tweet_sentiment_classifier.config.env_config import get_settings


class FormSubmissionRequest(BaseModel):
    feedback_text: str
    respondent: str | None = "Anonymous"


class SearchRequest(BaseModel):
    query: str


class PredictRequest(BaseModel):
    text: str


Label = Literal["positive", "neutral", "negative"]


class CorrectionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=get_settings().MAX_TEXT_CHARS)
    predicted: Label | Literal["uncertain"]
    corrected: Label
    decision_source: str | None = Field(default=None, max_length=100)
