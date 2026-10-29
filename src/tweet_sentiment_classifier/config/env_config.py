import os
from pydantic_settings import BaseSettings

from functools import lru_cache

class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./monitoring.db"
    GROQ_API_KEY: str = ""
    HF_API_TOKEN: str = ""
    HF_TOKEN: str = ""
    LOG_LEVEL: str = "INFO"
    # local: trained models only | llm: LLM decides | hybrid: LLM only when local confidence is low
    INFERENCE_MODE: str = "local"
    HYBRID_CONFIDENCE_THRESHOLD: float = 0.75

    class Config:
        env_file = ".env"
        extra = "ignore"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
