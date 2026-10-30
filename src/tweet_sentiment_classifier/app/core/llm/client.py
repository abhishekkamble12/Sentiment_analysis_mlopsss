"""Cached, budgeted, validated LLM calls.

Every answer must be a JSON object that passes a Pydantic model; anything else is retried once and then
reported as a failure (None), never guessed. Identical requests (same provider, model, prompt version and
input) are answered from the database. Every attempt's tokens are recorded, and once today's total reaches
the daily budget no further call is made.
"""
import hashlib
import logging
from datetime import datetime
from typing import Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError
from sqlalchemy import func
from sqlalchemy.orm import Session

from tweet_sentiment_classifier.app.core.llm.providers import LLMError, LLMProvider
from tweet_sentiment_classifier.entity.models import LlmCall
from tweet_sentiment_classifier.utils.llm_safety import extract_json_object

logger = logging.getLogger("api")
M = TypeVar("M", bound=BaseModel)
ATTEMPTS = 2


class BudgetExceeded(RuntimeError):
    pass


class LLMClient:
    def __init__(self, provider: LLMProvider | None, daily_token_budget: int):
        self.provider = provider
        self.daily_token_budget = daily_token_budget

    @property
    def configured(self) -> bool:
        return self.provider is not None

    def tokens_today(self, db: Session, now: Optional[datetime] = None) -> int:
        start = (now or datetime.utcnow()).replace(hour=0, minute=0, second=0, microsecond=0)
        total = db.query(func.coalesce(func.sum(LlmCall.tokens_in + LlmCall.tokens_out), 0)).filter(
            LlmCall.created_at >= start).scalar()
        return int(total or 0)

    def call_json(self, db: Session, *, prompt_version: str, system: str, user: str, model_cls: Type[M],
                  max_tokens: int = 600, now: Optional[datetime] = None,
                  usage: Optional[dict] = None) -> Optional[M]:
        """A validated answer, or None when the provider is missing, fails twice or never returns valid JSON.
        Raises BudgetExceeded when today's token budget is used up (cached answers are still served)."""
        if self.provider is None:
            return None
        p = self.provider
        key = hashlib.sha256("\x1f".join((p.name, p.model, prompt_version, system, user)).encode()).hexdigest()
        hit = db.query(LlmCall).filter(LlmCall.key == key, LlmCall.ok.is_(True)).order_by(LlmCall.id.desc()).first()
        if hit is not None:
            try:
                return model_cls.model_validate(hit.response)
            except ValidationError:
                pass  # the schema changed since this was cached: ask again

        for attempt in range(ATTEMPTS):
            if self.tokens_today(db, now) >= self.daily_token_budget:
                raise BudgetExceeded(f"daily LLM token budget of {self.daily_token_budget} is used up")
            row = LlmCall(key=key, provider=p.name, model=p.model, prompt_version=prompt_version,
                          created_at=now or datetime.utcnow())
            try:
                reply = p.complete(system, user, max_tokens)
            except LLMError as e:
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
                db.add(row)
                db.commit()
                continue
            row.tokens_in, row.tokens_out = reply.tokens_in, reply.tokens_out
            if usage is not None:  # lets a caller (the agent) cap what one session spends
                usage["in"] = usage.get("in", 0) + reply.tokens_in
                usage["out"] = usage.get("out", 0) + reply.tokens_out
            parsed = extract_json_object(reply.text)
            try:
                result = model_cls.model_validate(parsed)
            except ValidationError:
                logger.warning(f"LLM answer failed validation (attempt {attempt + 1})")
                db.add(row)
                db.commit()
                continue
            row.ok, row.response = True, result.model_dump()
            db.add(row)
            db.commit()
            return result
        return None
