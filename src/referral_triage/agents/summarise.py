from referral_triage.graph.state import ReferralState
from referral_triage.llm import get_llm

SYSTEM = """Summarise this masked GP referral note for specialist triage.
Include: age, main problem, duration, severity and impact, investigations mentioned
(with dates), treatments tried, and any worrying symptoms mentioned.
Do not add facts. Keep tokens such as <PERSON_1> exactly as written.
If a Critic note is provided, address it."""


async def summarise(state: ReferralState) -> dict:
    critic_note = state.get("critic", {}).get("notes", "") if state.get("rework_count") else ""
    msg = f"NOTE:\n{state['masked_note']}"
    if critic_note:
        msg += f"\n\nCRITIC FEEDBACK FROM LAST ROUND:\n{critic_note}"
    reply = await get_llm().ainvoke([("system", SYSTEM), ("human", msg)])
    return {"summary": reply.content}
