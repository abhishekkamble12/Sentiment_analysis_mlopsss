"""API keys for the endpoints that write, collect or spend LLM tokens.

A key is random (256 bits), shown once and stored only as its SHA-256, so a database leak does not leak keys.
The static INGEST_TOKEN setting keeps working next to them (handy for a cron job).

    python -m tweet_sentiment_classifier.app.core.auth create "ci cron"
    python -m tweet_sentiment_classifier.app.core.auth list
    python -m tweet_sentiment_classifier.app.core.auth revoke 3
"""
import hashlib
import hmac
import secrets
import sys
from typing import Iterable, Literal

from sqlalchemy.orm import Session

from tweet_sentiment_classifier.entity.models import ApiKey


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def create_key(db: Session, name: str) -> str:
    key = "sk_" + secrets.token_urlsafe(32)
    db.add(ApiKey(name=name.strip()[:100] or "unnamed", key_hash=hash_key(key)))
    db.commit()
    return key


def revoke_key(db: Session, key_id: int) -> bool:
    row = db.get(ApiKey, key_id)
    if row is None:
        return False
    row.revoked = True
    db.commit()
    return True


def authorize(db: Session, presented: Iterable[str], static_token: str) -> Literal["ok", "denied", "disabled"]:
    """"disabled" = nothing is configured to accept (no static token, no active key): protected endpoints stay shut."""
    values = [p for p in presented if p]
    has_keys = db.query(ApiKey.id).filter(ApiKey.revoked.is_(False)).first() is not None
    if not static_token and not has_keys:
        return "disabled"
    for value in values:
        if static_token and hmac.compare_digest(value.encode(), static_token.encode()):
            return "ok"
        if db.query(ApiKey.id).filter(ApiKey.key_hash == hash_key(value), ApiKey.revoked.is_(False)).first():
            return "ok"
    return "denied"


def main(argv=None) -> int:
    from tweet_sentiment_classifier.config.database import Base, SessionLocal, engine
    args = list(sys.argv[1:] if argv is None else argv)
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        if args[:1] == ["create"] and len(args) >= 2:
            print("API key (shown once, store it now):\n" + create_key(db, " ".join(args[1:])))
            return 0
        if args[:1] == ["list"]:
            for k in db.query(ApiKey).order_by(ApiKey.id):
                print(f"{k.id}\t{k.name}\t{'revoked' if k.revoked else 'active'}\t{k.created_at:%Y-%m-%d}")
            return 0
        if args[:1] == ["revoke"] and len(args) == 2 and args[1].isdigit():
            ok = revoke_key(db, int(args[1]))
            print("revoked" if ok else "no such key")
            return 0 if ok else 1
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
