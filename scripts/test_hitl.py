import os
os.environ["AUDIT_DISABLED"] = "1"          # no Postgres needed for this test

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from referral_triage.graph.nodes import finalise, gp_review
from referral_triage.graph.state import ReferralState


def fake_recommend(state):
    return {"recommendation": {"decision": "MISSING_INFO",
                               "rationale": "No X-ray in the last 12 months.",
                               "draft_letter": "Dear team, re <PERSON_1> ..."}}


g = StateGraph(ReferralState)
g.add_node("recommend", fake_recommend)
g.add_node("gp_review", gp_review)
g.add_node("finalise", finalise)
g.add_edge(START, "recommend")
g.add_edge("recommend", "gp_review")
g.add_edge("gp_review", "finalise")
g.add_edge("finalise", END)
app = g.compile(checkpointer=MemorySaver())

cfg = {"configurable": {"thread_id": "test-001"}}
start = {"case_id": "TEST-001", "token_map": {"<PERSON_1>": "Aroha Ngata"}, "findings": []}

# 1) Runs until the pause
result = app.invoke(start, cfg)
payload = result["__interrupt__"][0].value
print("PAUSED. GP sees:", payload["draft_letter_reidentified"])
print("Next step waiting:", app.get_state(cfg).next)

# 2) The GP approves - same thread_id, resumes where it stopped
result = app.invoke(Command(resume={"action": "approve", "reason": ""}), cfg)
print("FINAL LETTER:", result["final_letter"])