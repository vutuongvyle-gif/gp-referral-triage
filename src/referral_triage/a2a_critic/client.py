"""A2A client for the Critic service: discover it by its Agent Card, send JSON, read the verdict."""
import json
import uuid

import httpx
from a2a import types as a2a_types
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory

from referral_triage.config import settings

A2A_JSON = "application/json"


async def call_critic(payload: dict) -> dict:
    base = f"http://{settings.critic_host}:{settings.critic_port}"
    async with httpx.AsyncClient(timeout=180) as http:
        card = await A2ACardResolver(http, base_url=base).get_agent_card()
        if not any(s.id == "critic" for s in card.skills):
            raise RuntimeError(f"Agent '{card.name}' does not advertise the 'critic' skill")
        client = ClientFactory(ClientConfig(httpx_client=http, streaming=False)).create(card)

        part = a2a_types.Part()
        part.text, part.media_type = json.dumps(payload), A2A_JSON
        msg = a2a_types.Message(role=a2a_types.Role.ROLE_USER, message_id=str(uuid.uuid4()),
                                context_id=payload["case_id"])
        msg.parts.append(part)
        req = a2a_types.SendMessageRequest(
            message=msg,
            configuration=a2a_types.SendMessageConfiguration(accepted_output_modes=[A2A_JSON]))

        async for resp in client.send_message(req):
            if not resp.HasField("task"):
                continue
            task = resp.task
            if task.status.state == a2a_types.TaskState.TASK_STATE_FAILED:
                detail = " ".join(p.text for p in task.status.message.parts if p.HasField("text"))
                raise RuntimeError(f"Critic task failed: {detail or 'no detail'}")
            if task.status.state == a2a_types.TaskState.TASK_STATE_COMPLETED:
                for art in task.artifacts:
                    for p in art.parts:
                        if p.HasField("text"):
                            return json.loads(p.text)
    raise RuntimeError("Critic returned no artifact")
