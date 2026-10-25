from langgraph.graph import END, START, StateGraph

from referral_triage.agents.alternatives import alternatives_agent
from referral_triage.agents.completeness import completeness_agent
from referral_triage.agents.criteria import criteria_agent
from referral_triage.agents.red_flags import red_flag_agent
from referral_triage.agents.recommend import recommend
from referral_triage.agents.summarise import summarise
from referral_triage.graph.nodes import critic_node, finalise, gp_review, redact
from referral_triage.graph.state import ReferralState

SPECIALISTS = ["criteria", "completeness", "red_flags", "alternatives"]
MAX_REWORK = 2


def after_critic(state: ReferralState) -> str:
    if state["critic"].get("verdict") == "REWORK" and state.get("rework_count", 0) <= MAX_REWORK:
        return "summarise"
    return "recommend"


def build_graph(checkpointer):
    g = StateGraph(ReferralState)
    g.add_node("redact", redact)
    g.add_node("summarise", summarise)
    g.add_node("criteria", criteria_agent)
    g.add_node("completeness", completeness_agent)
    g.add_node("red_flags", red_flag_agent)
    g.add_node("alternatives", alternatives_agent)
    g.add_node("critic", critic_node)          # calls the Critic service over A2A
    g.add_node("recommend", recommend)
    g.add_node("gp_review", gp_review)         # interrupt() lives here
    g.add_node("finalise", finalise)

    g.add_edge(START, "redact")
    g.add_edge("redact", "summarise")
    for name in SPECIALISTS:                   # fan-out
        g.add_edge("summarise", name)
    g.add_edge(SPECIALISTS, "critic")          # fan-in: waits for all four
    g.add_conditional_edges("critic", after_critic, ["summarise", "recommend"])
    g.add_edge("recommend", "gp_review")
    g.add_edge("gp_review", "finalise")
    g.add_edge("finalise", END)
    return g.compile(checkpointer=checkpointer)