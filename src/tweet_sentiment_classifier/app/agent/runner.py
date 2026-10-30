"""The agent loop: choose a tool or answer, observe, repeat, then verify the citations.

Guardrails (all enforced here, none left to the model): a step limit, a per-tool call limit, a token limit and a
wall-clock limit for the session, tool arguments validated by Pydantic, tool results passed back as untrusted
data, and a citation check that removes any cited id no tool returned in this session. The whole trace is saved
after every step so a client can watch progress by polling.

shortcut: a plain loop over the JSON-protocol LLM client instead of LangGraph; replace it if branching plans or
parallel tool calls are needed. Cost is reported as tokens; there is no price table.
"""
import json
import logging
import re
import time
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError, model_validator
from sqlalchemy.orm import Session

from tweet_sentiment_classifier.app import prompts
from tweet_sentiment_classifier.app.agent.tools import ToolError, build_tools
from tweet_sentiment_classifier.app.core.llm.client import BudgetExceeded
from tweet_sentiment_classifier.entity.models import AgentRun, Topic
from tweet_sentiment_classifier.utils.llm_safety import sanitize_untrusted, wrap_untrusted

logger = logging.getLogger("api")
CALL_LIMITS = {"fetch_fresh": 1, "classify": 2}  # every other tool: DEFAULT_CALL_LIMIT
DEFAULT_CALL_LIMIT = 4
OBSERVATION_CHARS = 3000


class AgentStep(BaseModel):
    action: Literal["tool", "final"]
    tool: Optional[str] = Field(default=None, max_length=40)
    args: Dict[str, Any] = Field(default_factory=dict)
    answer: Optional[str] = Field(default=None, max_length=6000)
    citations: List[int] = Field(default_factory=list, max_length=30)

    @model_validator(mode="after")
    def _complete(self):
        if self.action == "tool" and not self.tool:
            raise ValueError("a tool action needs a tool name")
        if self.action == "final" and not (self.answer or "").strip():
            raise ValueError("a final action needs an answer")
        return self


def verify_answer(answer: str, citations: List[int], seen: Dict[int, Dict[str, Any]],
                  tools_called: int = 0) -> Dict[str, Any]:
    """Keep only citations of posts a tool returned in this session; strip the others from the text."""
    cited = set(citations) | {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
    valid, invalid = sorted(cited & seen.keys()), sorted(cited - seen.keys())
    text = re.sub(r"\[(\d+)\]", lambda m: m.group(0) if int(m.group(1)) in seen else "", answer)
    return {"text": re.sub(r"[ \t]{2,}", " ", text).strip(), "citations": valid, "removed_citations": invalid,
            "sources": [seen[i] for i in valid],
            # verified = at least one real citation, or the agent looked (a tool ran) and found no posts to cite
            "verified": bool(valid) or (tools_called > 0 and not seen and not cited)}


def _render(run: AgentRun, topic: Optional[Topic], observations: List[str], step: int, max_steps: int) -> str:
    lines = [f"Mode: {run.kind}"]
    if topic is not None:
        lines.append(f"Topic id: {topic.id} (name: {sanitize_untrusted(topic.name, 100)})")
    lines.append("Question:\n" + wrap_untrusted("query", run.question, 600))
    if run.kind == "report":
        lines.append(prompts.AGENT_REPORT_FORMAT)
    lines.append(f"Step {step + 1} of {max_steps}.")
    lines.append("Observations so far:\n" + ("\n".join(observations) if observations else "(none yet)"))
    if step == max_steps - 1:
        lines.append('This is your last step: reply with {"action": "final", ...} now.')
    return "\n".join(lines)


def _collect_posts(result: Any, seen: Dict[int, Dict[str, Any]]) -> None:
    """Remember every post a tool result contains: only these may be cited."""
    if isinstance(result, dict):
        if isinstance(result.get("id"), int) and "text" in result and "sentiment" in result:
            seen[result["id"]] = {k: result.get(k) for k in ("id", "text", "sentiment", "url", "source")}
        for value in result.values():
            _collect_posts(value, seen)
    elif isinstance(result, list):
        for item in result:
            _collect_posts(item, seen)


def run_agent(db: Session, c, run: AgentRun) -> AgentRun:
    """Execute a prepared AgentRun (status "running") to completion and persist the outcome."""
    started = time.monotonic()
    settings, client = c.settings, c.llm
    topic = db.get(Topic, run.topic_id) if run.topic_id else None
    tools = build_tools(c, db)
    system = prompts.AGENT_SYSTEM.replace("{tools}", "\n".join(f"- {t.signature}" for t in tools.values()))
    usage: Dict[str, int] = {"in": 0, "out": 0}
    steps: List[Dict[str, Any]] = []
    observations: List[str] = []
    seen: Dict[int, Dict[str, Any]] = {}
    calls: Counter = Counter()

    def save(status: str = "running", error: Optional[str] = None, answer: Optional[dict] = None) -> AgentRun:
        run.steps, run.status, run.error, run.answer = list(steps), status, error, answer
        run.tokens_in, run.tokens_out = usage["in"], usage["out"]
        if status != "running":
            run.finished_at = datetime.utcnow()
            run.latency_ms = round((time.monotonic() - started) * 1000, 1)
        db.commit()
        return run

    for step in range(settings.AGENT_MAX_STEPS):
        if usage["in"] + usage["out"] >= settings.AGENT_MAX_TOKENS:
            return save("limit_reached", f"the session used its {settings.AGENT_MAX_TOKENS}-token limit")
        if time.monotonic() - started > settings.AGENT_TIMEOUT_SECONDS:
            return save("limit_reached", f"the session ran longer than {settings.AGENT_TIMEOUT_SECONDS} s")
        try:
            decision = client.call_json(db, prompt_version=prompts.AGENT_VERSION, system=system,
                                        user=_render(run, topic, observations, step, settings.AGENT_MAX_STEPS),
                                        model_cls=AgentStep, max_tokens=1500, usage=usage)
        except BudgetExceeded as e:
            return save("failed", str(e))
        if decision is None:
            return save("failed", "the model did not return a valid step")

        if decision.action == "final":
            steps.append({"step": step + 1, "action": "final"})
            return save("done", answer=verify_answer(decision.answer, decision.citations, seen, sum(calls.values())))

        record: Dict[str, Any] = {"step": step + 1, "action": "tool", "tool": decision.tool, "args": decision.args}
        tool = tools.get(decision.tool)
        try:
            if tool is None:
                raise ToolError(f"unknown tool '{decision.tool}'; the tools are {', '.join(tools)}")
            if calls[tool.name] >= CALL_LIMITS.get(tool.name, DEFAULT_CALL_LIMIT):
                raise ToolError(f"{tool.name} was already called as often as allowed in one session")
            try:
                args = tool.args.model_validate(decision.args)
            except ValidationError as e:
                raise ToolError("invalid arguments: " + "; ".join(
                    f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()[:3])) from None
            calls[tool.name] += 1
            result: Dict[str, Any] = tool.fn(args)
            _collect_posts(result, seen)
        except ToolError as e:
            result = {"error": str(e)}
        except Exception as e:  # a tool bug must not end the session: the model is told and can adapt
            logger.warning(f"agent tool {decision.tool} failed: {e}")
            db.rollback()
            result = {"error": f"{decision.tool} failed"}
        text = json.dumps(result, default=str, separators=(",", ":"))
        record["result"] = text[:300]
        steps.append(record)
        observations.append(f"[step {step + 1}] {decision.tool}({json.dumps(decision.args)}):\n"
                            + wrap_untrusted("data", text, OBSERVATION_CHARS))
        save()

    return save("incomplete", f"no answer after {settings.AGENT_MAX_STEPS} steps")


def execute(session_factory, c, run_id: int) -> None:
    """Background entry point: its own database session, and nothing escapes (a crash is recorded on the run)."""
    with session_factory() as db:
        run = db.get(AgentRun, run_id)
        try:
            run_agent(db, c, run)
        except Exception as e:
            logger.exception("agent run crashed")
            db.rollback()
            run = db.get(AgentRun, run_id)
            run.status, run.error, run.finished_at = "failed", f"internal error: {type(e).__name__}", datetime.utcnow()
            db.commit()
