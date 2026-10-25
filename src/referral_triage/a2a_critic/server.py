"""Critic A2A service. Run: make critic"""
import json
from typing import Literal

import uvicorn
from a2a import types as a2a_types
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from pydantic import BaseModel
from starlette.applications import Starlette

from referral_triage.config import settings
from referral_triage.llm import get_llm

A2A_JSON = "application/json"
HOST, PORT = settings.critic_host, settings.critic_port

CRITIC_PROMPT = """You are an independent clinical-safety reviewer.
Given masked referral text and the specialist findings, check:
1. Every finding's evidence actually appears in the text or tool outputs.
2. No red-flag symptom in the text was missed.
3. The findings do not contradict each other.
Return REWORK if any check fails, otherwise PASS."""


class CriticVerdict(BaseModel):
    verdict: Literal["PASS", "REWORK"]
    ungrounded_findings: list[str]
    missed_red_flags: list[str]
    notes: str


def json_part(payload) -> a2a_types.Part:
    part = a2a_types.Part()
    part.text = payload if isinstance(payload, str) else json.dumps(payload)
    part.media_type = A2A_JSON
    return part


async def review(payload: dict) -> dict:
    llm = get_llm().with_structured_output(CriticVerdict)
    verdict = await llm.ainvoke([("system", CRITIC_PROMPT),
                                 ("human", json.dumps(payload))])
    return verdict.model_dump()


class CriticExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if not context.current_task:
            await event_queue.enqueue_event(a2a_types.Task(
                id=context.task_id, context_id=context.context_id,
                status=a2a_types.TaskStatus(state=a2a_types.TaskState.TASK_STATE_SUBMITTED)))
        await updater.start_work()
        try:
            payload = json.loads(context.get_user_input() or "{}")
            await updater.add_artifact([json_part(await review(payload))], name="critic-result")
            await updater.complete()
        except Exception as exc:
            await updater.failed(updater.new_agent_message(
                [json_part(f"{type(exc).__name__}: {exc}")]))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        await TaskUpdater(event_queue, context.task_id, context.context_id).cancel()


def build_app() -> Starlette:
    card = a2a_types.AgentCard(
        name="referral-critic",
        description="Independent checker for GP referral triage findings.",
        version="0.1.0",
        default_input_modes=[A2A_JSON], default_output_modes=[A2A_JSON],
        capabilities=a2a_types.AgentCapabilities(streaming=False, push_notifications=False),
        supported_interfaces=[a2a_types.AgentInterface(
            url=f"http://{HOST}:{PORT}/", protocol_binding="JSONRPC", protocol_version="1.0")],
        skills=[a2a_types.AgentSkill(
            id="critic", name="Referral critic",
            description="Checks findings are grounded in evidence and no red flag was missed.",
            tags=["referral", "qa", "critic"], input_modes=[A2A_JSON], output_modes=[A2A_JSON])],
    )
    handler = DefaultRequestHandler(agent_executor=CriticExecutor(),
                                    task_store=InMemoryTaskStore(), agent_card=card)
    return Starlette(routes=create_agent_card_routes(card)
                     + create_jsonrpc_routes(handler, rpc_url="/"))


if __name__ == "__main__":
    uvicorn.run(build_app(), host=HOST, port=PORT, log_level="warning")