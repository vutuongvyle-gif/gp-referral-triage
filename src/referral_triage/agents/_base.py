"""Shared pattern for specialist agents: context in, validated findings out."""
from pydantic import BaseModel

from referral_triage.graph.state import Finding, ReferralState
from referral_triage.llm import get_llm


class AgentResult(BaseModel):
    findings: list[Finding]


async def run_agent(state: ReferralState, agent: str, system: str, context: str) -> dict:
    llm = get_llm().with_structured_output(AgentResult)
    out: AgentResult = await llm.ainvoke([
        ("system", system),
        ("human", f"{context}\n\nREFERRAL SUMMARY:\n{state.get('summary', '')}"),
    ])
    rnd = state.get("rework_count", 0)
    return {"findings": [f.model_dump() | {"agent": agent, "round": rnd} for f in out.findings]}
