import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

Decision = Literal["READY_TO_SEND", "MISSING_INFO", "CONSIDER_ALTERNATIVE", "URGENT_ESCALATE"]


class Finding(BaseModel):
    agent: str = ""
    claim: str
    evidence: str = Field(description="Exact quote from the note, or a FHIR resource id")
    criterion_id: str | None = Field(default=None, description="e.g. ORTHO-01")
    severity: Literal["info", "gap", "red_flag"] = "info"
    round: int = 0


class ReferralState(TypedDict, total=False):
    case_id: str
    patient_ref: str          # FHIR logical id - not an identifier like the NHI
    specialty: str
    raw_note: str             # original text: never sent to any LLM
    masked_note: str          # PII replaced by tokens such as <NZ_NHI_1>
    token_map: dict           # local only, used to re-identify the final letter
    summary: str
    findings: Annotated[list[dict], operator.add]
    critic: dict
    rework_count: int
    recommendation: dict      # {"decision": Decision, "rationale": str, "draft_letter": str}
    gp_decision: dict         # {"action": "approve|edit|alternative|reject", ...}
    final_letter: str