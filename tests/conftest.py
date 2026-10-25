"""Shared test fixtures: fake models so unit tests run in seconds with no Ollama."""
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

# Every module that does `from referral_triage.llm import get_llm`.
# Add new agent modules here as you create them.
LLM_MODULES = [
    "referral_triage.agents.summarise",
    "referral_triage.agents._base",
    "referral_triage.agents.criteria",
    "referral_triage.agents.recommend",
]


def _patch_everywhere(monkeypatch, model):
    for mod in LLM_MODULES:
        monkeypatch.setattr(f"{mod}.get_llm", lambda role="agent": model)


@pytest.fixture
def fake_text_llm(monkeypatch):
    """For plain-text nodes. Usage: fake_text_llm("reply 1", "reply 2")"""
    def install(*replies: str):
        model = GenericFakeChatModel(messages=iter([AIMessage(content=r) for r in replies]))
        _patch_everywhere(monkeypatch, model)
        return model
    return install


class StructuredStub:
    """Stands in for get_llm().with_structured_output(Schema).
    Each call returns the next scripted dict, validated against the real Schema,
    so a reply that breaks the schema fails the test just as a bad model reply would."""

    def __init__(self, replies: list[dict]):
        self.replies = iter(replies)
        self.prompts = []           # what the node sent, so tests can inspect it

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    async def ainvoke(self, messages):
        self.prompts.append(messages)
        return self.schema.model_validate(next(self.replies))

    def invoke(self, messages):
        self.prompts.append(messages)
        return self.schema.model_validate(next(self.replies))


@pytest.fixture
def fake_structured_llm(monkeypatch):
    """For agents using with_structured_output. Usage: fake_structured_llm({...}, {...})"""
    def install(*replies: dict):
        stub = StructuredStub(list(replies))
        _patch_everywhere(monkeypatch, stub)
        return stub
    return install