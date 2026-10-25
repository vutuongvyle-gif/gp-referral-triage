"""Graph nodes that are not specialist agents: redaction, Critic call, GP review, finalise."""
import os

from langgraph.types import interrupt

from referral_triage.graph.state import ReferralState

from referral_triage.a2a_critic.client import call_critic
from referral_triage.mcp_servers.client import CALL_LOG

VALID_ACTIONS = {"approve", "edit", "alternative", "reject"}


def _unmask(text: str, token_map: dict) -> str:
    """Put real identifiers back. Plain string replacement - no model involved."""
    for token, original in (token_map or {}).items():
        text = text.replace(token, original)
    return text


# ---------------------------------------------------------------- redact
# TEMP until section 8: replace the body with the Presidio version shown below.
def redact(state: ReferralState) -> dict:
    return {"masked_note": state["raw_note"], "token_map": {}}

# Section 8 version:
# _analyzer = None
# def redact(state: ReferralState) -> dict:
#     global _analyzer
#     from referral_triage.privacy.masking import build_analyzer, mask
#     if _analyzer is None:
#         _analyzer = build_analyzer()
#     masked, tokens = mask(state["raw_note"], _analyzer,
#                           known_names=state.get("patient_names"))
#     return {"masked_note": masked, "token_map": tokens}


# ---------------------------------------------------------------- critic
MAX_TOOL_CHARS = 2000   # keep the payload small enough for a local model


async def critic_node(state: ReferralState) -> dict:
    """Send the round's evidence to the independent Critic service over A2A."""
    rnd = state.get("rework_count", 0)
    tool_outputs = [{"tool": c["tool"], "output": c["output"][:MAX_TOOL_CHARS]}
                    for c in CALL_LOG if c["case_id"] == state["case_id"]]
    verdict = await call_critic({
        "case_id": state["case_id"],
        "masked_note": state["masked_note"],
        "findings": [f for f in state.get("findings", []) if f.get("round", 0) == rnd],
        "tool_outputs": tool_outputs,
    })
    return {"critic": verdict,
            "rework_count": rnd + (verdict.get("verdict") == "REWORK")}


# ---------------------------------------------------------------- GP review (native HITL)
def gp_review(state: ReferralState) -> dict:
    """Pause for the GP. Everything before interrupt() re-runs on resume, so keep it pure."""
    rec = state["recommendation"]
    decision = interrupt({
        "case_id": state["case_id"],
        "recommendation": rec,
        "findings": state.get("findings", []),
        "critic": state.get("critic", {}),
        "draft_letter_reidentified": _unmask(rec.get("draft_letter", ""),
                                             state.get("token_map", {})),
    })
    if decision.get("action") not in VALID_ACTIONS:
        raise ValueError(f"Unknown GP action: {decision.get('action')!r}")
    return {"gp_decision": decision}


# ---------------------------------------------------------------- finalise
def _log_decision(state: ReferralState, final_letter: str) -> None:
    """Write the GP's decision to the audit table (needs Postgres from section 10)."""
    import psycopg

    from referral_triage.config import settings
    with psycopg.connect(settings.database_url) as conn:
        conn.execute(
            "INSERT INTO gp_decisions (case_id, ai_decision, gp_action, reason)"
            " VALUES (%s, %s, %s, %s)",
            (state["case_id"], state["recommendation"]["decision"],
             state["gp_decision"]["action"], state["gp_decision"].get("reason", "")),
        )


def finalise(state: ReferralState) -> dict:
    d, rec = state["gp_decision"], state["recommendation"]
    if d["action"] == "approve":
        letter = _unmask(rec.get("draft_letter", ""), state.get("token_map", {}))
    elif d["action"] in ("edit", "alternative"):
        letter = d.get("letter", "")      # GP edited the re-identified text already
    else:                                 # reject
        letter = ""
    if os.getenv("AUDIT_DISABLED") != "1":
        _log_decision(state, letter)
    return {"final_letter": letter}