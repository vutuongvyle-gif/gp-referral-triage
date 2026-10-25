import asyncio

import pytest

from referral_triage.agents.criteria import criteria_agent
from referral_triage.agents.summarise import summarise


def test_summarise_returns_model_text(fake_text_llm):
    fake_text_llm("Summary: knee pain 8 months")
    out = asyncio.run(summarise({"masked_note": "<PERSON_1>, knee pain", "rework_count": 0}))
    assert out["summary"] == "Summary: knee pain 8 months"


def test_criteria_agent_tags_and_rounds_findings(fake_structured_llm, monkeypatch):
    stub = fake_structured_llm({
        "meets_criteria": False,
        "findings": [{"claim": "No weight-bearing X-ray in 12 months",
                      "evidence": "not found in records",
                      "criterion_id": "ORTHO-01", "severity": "gap"}],
    })

    async def fake_tool(name, args, case_id=None):        # no MCP server needed
        return "[ORTHO-01 | Required before referral] Weight-bearing X-ray within 12 months"
    monkeypatch.setattr("referral_triage.agents.criteria.call_tool", fake_tool)

    state = {"case_id": "T-1", "specialty": "orthopaedics", "summary": "knee pain",
             "rework_count": 1}
    out = asyncio.run(criteria_agent(state))

    f = out["findings"][0]
    assert f["agent"] == "criteria"            # the agent stamps its own name
    assert f["round"] == 1                     # and the current rework round
    assert f["severity"] == "gap"
    assert "ORTHO-01" in str(stub.prompts[0])  # the criteria text reached the prompt


def test_invalid_model_output_is_rejected(fake_structured_llm, monkeypatch):
    fake_structured_llm({"meets_criteria": True,
                         "findings": [{"claim": "x", "evidence": "y", "severity": "serious"}]})

    async def fake_tool(name, args, case_id=None):
        return ""
    monkeypatch.setattr("referral_triage.agents.criteria.call_tool", fake_tool)

    with pytest.raises(Exception):             # pydantic ValidationError
        asyncio.run(criteria_agent({"case_id": "T", "specialty": "o", "summary": "s"}))