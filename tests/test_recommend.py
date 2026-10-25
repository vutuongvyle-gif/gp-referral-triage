import asyncio

import pytest

from referral_triage.agents import recommend as rec_mod


class StubModel:
    """Pretends to be the LLM. Returns a fixed decision, or fails if it must not be called."""
    def __init__(self, decision=None):
        self.decision = decision

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    async def ainvoke(self, messages):
        if self.decision is None:
            raise AssertionError("The model must not be called for a red-flag case")
        return self.schema(decision=self.decision, rationale="stub", draft_letter="Dear team")


def run(state, stub, monkeypatch):
    monkeypatch.setattr(rec_mod, "get_llm", lambda role="agent": stub)
    return asyncio.run(rec_mod.recommend(state))["recommendation"]


def finding(severity, claim="x"):
    return {"agent": "t", "claim": claim, "evidence": "e", "severity": severity, "round": 0}


def test_red_flag_skips_model(monkeypatch):
    state = {"specialty": "orthopaedics", "rework_count": 0,
             "findings": [finding("red_flag", "Hot swollen joint with fever")]}
    rec = run(state, StubModel(decision=None), monkeypatch)
    assert rec["decision"] == "URGENT_ESCALATE"
    assert rec["overrides"] == ["red_flag_rule"]


def test_ready_with_gap_is_overridden(monkeypatch):
    state = {"specialty": "orthopaedics", "rework_count": 0,
             "findings": [finding("gap", "No X-ray in last 12 months")]}
    rec = run(state, StubModel("READY_TO_SEND"), monkeypatch)
    assert rec["decision"] == "MISSING_INFO"
    assert "gap_rule" in rec["overrides"]


def test_clean_case_keeps_model_decision(monkeypatch):
    state = {"specialty": "orthopaedics", "rework_count": 0,
             "findings": [finding("info", "Symptoms 8 months")]}
    rec = run(state, StubModel("READY_TO_SEND"), monkeypatch)
    assert rec["decision"] == "READY_TO_SEND"
    assert rec["overrides"] == []


def test_old_round_red_flag_is_ignored(monkeypatch):
    old = finding("red_flag") | {"round": 0}
    state = {"specialty": "orthopaedics", "rework_count": 1,
             "findings": [old, finding("info") | {"round": 1}]}
    rec = run(state, StubModel("READY_TO_SEND"), monkeypatch)
    assert rec["decision"] == "READY_TO_SEND"