"""Recommendation step: deterministic safety rules first, the model only for the rest."""
from typing import Literal

from pydantic import BaseModel, Field

from referral_triage.graph.state import ReferralState
from referral_triage.llm import get_llm

SYSTEM = """You are the Recommendation Agent in a New Zealand GP referral triage system.
You receive findings from specialist agents. Choose ONE decision:
- READY_TO_SEND: criteria met and no required information is missing.
- MISSING_INFO: the referral would likely be declined because something required
  is missing (an investigation, a trial of first-line treatment, duration of symptoms).
- CONSIDER_ALTERNATIVE: the case is below the acceptance threshold; recommend the
  alternative named in the findings (for example a specialist advice request).
Rules:
- Base the decision only on the findings provided. Do not add clinical facts.
- Draft a short, professional referral letter (or advice request) for the GP to edit.
- Keep every token such as <PERSON_1> or <NZ_NHI_1> exactly as written."""


class ModelRecommendation(BaseModel):
    decision: Literal["READY_TO_SEND", "MISSING_INFO", "CONSIDER_ALTERNATIVE"]
    rationale: str = Field(description="Two or three sentences citing the findings")
    draft_letter: str
    decline_reason_code: Literal["MISSING_INVESTIGATION", "BELOW_THRESHOLD",
                                 "NO_FIRST_LINE", "OTHER", "NONE"] = "NONE"


def latest_findings(state: ReferralState) -> list[dict]:
    rnd = state.get("rework_count", 0)
    return [f for f in state.get("findings", []) if f.get("round", 0) == rnd]


def format_findings(findings: list[dict]) -> str:
    return "\n".join(
        f"- [{f.get('agent')}|{f.get('severity')}|{f.get('criterion_id') or '-'}] "
        f"{f['claim']} (evidence: {f['evidence']})"
        for f in findings
    ) or "- No findings."


async def recommend(state: ReferralState) -> dict:
    findings = latest_findings(state)

    # Rule 1 - red flags always win. No model call.
    red = [f for f in findings if f.get("severity") == "red_flag"]
    if red:
        return {"recommendation": {
            "decision": "URGENT_ESCALATE",
            "rationale": "Red flag found: " + "; ".join(f["claim"] for f in red)
                         + ". Use the acute pathway, not a routine referral.",
            "draft_letter": "",
            "decline_reason_code": "NONE",
            "overrides": ["red_flag_rule"],
        }}

    # Model decides among the three non-urgent outcomes.
    llm = get_llm().with_structured_output(ModelRecommendation)
    out: ModelRecommendation = await llm.ainvoke([
        ("system", SYSTEM),
        ("human", f"SPECIALTY: {state['specialty']}\n\n"
                  f"REFERRAL SUMMARY:\n{state.get('summary', '')}\n\n"
                  f"FINDINGS:\n{format_findings(findings)}"),
    ])
    rec = out.model_dump() | {"overrides": []}

    # Rule 2 - "ready" is not allowed while a gap is unresolved.
    gaps = [f for f in findings if f.get("severity") == "gap"]
    if rec["decision"] == "READY_TO_SEND" and gaps:
        rec["decision"] = "MISSING_INFO"
        rec["rationale"] = ("Changed from READY_TO_SEND because these requirements are unmet: "
                            + "; ".join(f["claim"] for f in gaps) + ". " + rec["rationale"])
        rec["overrides"].append("gap_rule")
        if rec["decline_reason_code"] == "NONE":
            rec["decline_reason_code"] = "MISSING_INVESTIGATION"

    return {"recommendation": rec}