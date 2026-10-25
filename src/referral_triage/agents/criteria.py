from pydantic import BaseModel

from referral_triage.graph.state import Finding, ReferralState
from referral_triage.llm import get_llm
from referral_triage.mcp_servers.client import call_tool

SYSTEM = """You are the Criteria Agent in a New Zealand GP referral triage system.
Decide whether the referral meets the specialty's acceptance criteria.
Rules:
- Use ONLY the criteria text provided. If it does not cover the case, say so.
- Every finding must include evidence: an exact quote from the referral
  summary or a tool result, and the criterion id it relates to.
- Mark unmet requirements as severity "gap". Do not invent clinical facts."""


class CriteriaResult(BaseModel):
    meets_criteria: bool
    findings: list[Finding]


async def criteria_agent(state: ReferralState) -> dict:
    criteria = await call_tool(
        "retrieve_criteria", {"specialty": state["specialty"], "query": state["summary"]}
    )
    llm = get_llm().with_structured_output(CriteriaResult)
    result: CriteriaResult = await llm.ainvoke([
        ("system", SYSTEM),
        ("human", f"CRITERIA:\n{criteria}\n\nREFERRAL SUMMARY:\n{state['summary']}"),
    ])
    rnd = state.get("rework_count", 0)
    return {"findings": [f.model_dump() | {"agent": "criteria", "round": rnd}
                         for f in result.findings]}