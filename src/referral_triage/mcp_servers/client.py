"""MCP client used by the agents. Connects only when a tool is first called."""
from langchain_mcp_adapters.client import MultiServerMCPClient

from referral_triage.config import settings

_client = MultiServerMCPClient({
    "referral_tools": {
        "url": f"http://{settings.mcp_host}:{settings.mcp_port}/mcp",
        "transport": "streamable_http",
    }
})
_tools: dict | None = None
CALL_LOG: list[dict] = []          # DeepEval tool-correctness metrics read this


async def call_tool(name: str, args: dict, case_id: str | None = None) -> str:
    """Call one MCP tool by name and return its text output."""
    global _tools
    if _tools is None:
        _tools = {t.name: t for t in await _client.get_tools()}
    if name not in _tools:
        raise KeyError(f"Unknown MCP tool '{name}'. Available: {sorted(_tools)}")
    result = await _tools[name].ainvoke(args)
    text = result if isinstance(result, str) else "\n".join(
        b.get("text", "") if isinstance(b, dict) else str(b) for b in result)
    CALL_LOG.append({"case_id": case_id, "tool": name, "args": args, "output": text})
    return text