from pydantic import BaseModel


class FormSubmissionRequest(BaseModel):
    feedback_text: str
    respondent: str | None = "Anonymous"


class SearchRequest(BaseModel):
    query: str


class PredictRequest(BaseModel):
    text: str
