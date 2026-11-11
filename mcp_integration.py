"""
mcp_integration.py — Additive MCP client layer for AI Job Hunt OS.

This module does NOT modify any existing app behaviour. It provides:
  - Persistent background event loop for Streamlit compatibility
  - MCP server connection (stdio / Streamable HTTP / SSE)
  - Tool listing and calling
  - MCP tool schema → OpenAI function-calling conversion
  - A simple tool-calling agent loop using the existing Groq / Sea-Lion clients

Requires: pip install mcp
"""

from __future__ import annotations

import asyncio
import json
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Optional MCP SDK import — the app must keep working without it
# ---------------------------------------------------------------------------

try:
    from mcp import ClientSession, StdioServerParameters, types
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.client.sse import sse_client

    MCP_AVAILABLE = True
    MCP_IMPORT_ERROR = ""
except ImportError as exc:
    MCP_AVAILABLE = False
    MCP_IMPORT_ERROR = str(exc)

    ClientSession = Any  # type: ignore
    StdioServerParameters = Any  # type: ignore
    types = Any  # type: ignore
    stdio_client = None  # type: ignore
    streamablehttp_client = None  # type: ignore
    sse_client = None  # type: ignore


# ---------------------------------------------------------------------------
# 1. Persistent background event loop
# ---------------------------------------------------------------------------

_MCP_LOOP: Optional[asyncio.AbstractEventLoop] = None
_MCP_THREAD: Optional[threading.Thread] = None
_MCP_LOCK = threading.Lock()


def _ensure_mcp_loop() -> asyncio.AbstractEventLoop:
    """Create (once) and return a persistent background event loop."""
    global _MCP_LOOP, _MCP_THREAD

    with _MCP_LOCK:
        if _MCP_LOOP is None or _MCP_THREAD is None or not _MCP_THREAD.is_alive():
            _MCP_LOOP = asyncio.new_event_loop()
            _MCP_THREAD = threading.Thread(
                target=_MCP_LOOP.run_forever,
                daemon=True,
                name="mcp-event-loop",
            )
            _MCP_THREAD.start()
        return _MCP_LOOP


def run_mcp_coro(coro, timeout: float = 90.0):
    """Submit an async coroutine to the background loop and block for the result."""
    loop = _ensure_mcp_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=timeout)


# ---------------------------------------------------------------------------
# 2. Server configuration
# ---------------------------------------------------------------------------

@dataclass
class MCPServerConfig:
    """
    Configuration for one MCP server.

    - stdio:   command + args (e.g. command="uvx", args=["job-offer-scraper-mcp"])
    - http:    url ending in /mcp (Streamable HTTP)
    - sse:     url ending in /sse
    """

    name: str
    transport: str = "stdio"  # "stdio" | "http" | "sse"
    command: str = ""
    args: list = field(default_factory=list)
    url: str = ""
    env: Optional[dict] = None
    cwd: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "transport": self.transport,
            "command": self.command,
            "args": list(self.args),
            "url": self.url,
            "env": dict(self.env or {}),
            "cwd": self.cwd,
        }


# ---------------------------------------------------------------------------
# 3. Connection / tool operations
# ---------------------------------------------------------------------------

async def _open_session(config: MCPServerConfig, stack: AsyncExitStack):
    """Open an MCP ClientSession for the given server config."""
    if config.transport == "stdio":
        if not config.command:
            raise ValueError("stdio transport requires a 'command'.")
        params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=config.env,
            cwd=config.cwd,
        )
        read, write = await stack.enter_async_context(stdio_client(params))
    elif config.transport == "http":
        if not config.url:
            raise ValueError("http transport requires a 'url'.")
        read, write, _ = await stack.enter_async_context(
            streamablehttp_client(config.url)
        )
    elif config.transport == "sse":
        if not config.url:
            raise ValueError("sse transport requires a 'url'.")
        read, write = await stack.enter_async_context(sse_client(config.url))
    else:
        raise ValueError(f"Unsupported transport: {config.transport}")

    session = await stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    return session


async def _list_tools_async(config: MCPServerConfig):
    async with AsyncExitStack() as stack:
        session = await _open_session(config, stack)
        result = await session.list_tools()
        return result.tools


async def _call_tool_async(config: MCPServerConfig, tool_name: str, arguments: dict):
    async with AsyncExitStack() as stack:
        session = await _open_session(config, stack)
        return await session.call_tool(tool_name, arguments)


async def _list_resources_async(config: MCPServerConfig):
    async with AsyncExitStack() as stack:
        session = await _open_session(config, stack)
        result = await session.list_resources()
        return result.resources


# ---------------------------------------------------------------------------
# 4. Synchronous wrappers
# ---------------------------------------------------------------------------

def list_mcp_tools(config: MCPServerConfig) -> list:
    """Return the list of MCP Tool objects exposed by the server."""
    if not MCP_AVAILABLE:
        raise RuntimeError(f"MCP SDK not installed: {MCP_IMPORT_ERROR}")
    return run_mcp_coro(_list_tools_async(config), timeout=60)


def call_mcp_tool(config: MCPServerConfig, tool_name: str, arguments: dict) -> dict:
    """
    Call an MCP tool and normalise the result to a plain dict:
      {
        "ok": bool,
        "is_error": bool,
        "text": str,
        "structured": dict|None,
        "raw_content": list,
      }
    """
    if not MCP_AVAILABLE:
        raise RuntimeError(f"MCP SDK not installed: {MCP_IMPORT_ERROR}")

    result = run_mcp_coro(
        _call_tool_async(config, tool_name, arguments), timeout=120
    )

    text_parts: list = []
    raw_content: list = []

    for content in getattr(result, "content", []) or []:
        if MCP_AVAILABLE and isinstance(content, types.TextContent):
            text_parts.append(content.text)
            raw_content.append({"type": "text", "text": content.text})
        elif MCP_AVAILABLE and isinstance(content, types.ImageContent):
            raw_content.append({
                "type": "image",
                "mimeType": getattr(content, "mimeType", ""),
                "bytes": len(getattr(content, "data", b"") or b""),
            })
        elif MCP_AVAILABLE and isinstance(content, types.EmbeddedResource):
            raw_content.append({"type": "resource", "uri": str(getattr(content, "resource", ""))})
        else:
            raw_content.append({"type": type(content).__name__})

    structured = getattr(result, "structuredContent", None)
    is_error = bool(getattr(result, "isError", False))

    return {
        "ok": not is_error,
        "is_error": is_error,
        "text": "\n".join(text_parts).strip(),
        "structured": structured,
        "raw_content": raw_content,
    }


def list_mcp_resources(config: MCPServerConfig) -> list:
    """Return resources exposed by the server (optional feature)."""
    if not MCP_AVAILABLE:
        raise RuntimeError(f"MCP SDK not installed: {MCP_IMPORT_ERROR}")
    return run_mcp_coro(_list_resources_async(config), timeout=60)


# ---------------------------------------------------------------------------
# 5. MCP Tool → OpenAI function-calling schema
# ---------------------------------------------------------------------------

def mcp_tool_to_openai(tool) -> dict:
    """Convert an MCP Tool object to an OpenAI ChatCompletionTool dict."""
    schema = getattr(tool, "inputSchema", None) or {
        "type": "object",
        "properties": {},
    }
    if "type" not in schema:
        schema = {"type": "object", **schema}

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": (getattr(tool, "description", "") or "").strip(),
            "parameters": schema,
        },
    }


def mcp_tools_to_openai(tools) -> list:
    return [mcp_tool_to_openai(t) for t in tools]


# ---------------------------------------------------------------------------
# 6. Simple tool-calling agent loop (Groq / OpenAI-compatible)
# ---------------------------------------------------------------------------

def run_tool_agent(
    prompt: str,
    server_config: MCPServerConfig,
    chat_client,
    model_id: str,
    max_turns: int = 6,
    temperature: float = 0.4,
) -> dict:
    """
    Run a tool-calling loop against an MCP server.

    `chat_client` must be an OpenAI-compatible client with
    `.chat.completions.create(...)` (Groq's client and the OpenAI client both
    satisfy this).

    Returns:
      {
        "answer": str,
        "turns": int,
        "tool_calls": list[dict],
        "error": str|None,
      }
    """
    if not MCP_AVAILABLE:
        return {
            "answer": "",
            "turns": 0,
            "tool_calls": [],
            "error": f"MCP SDK not installed: {MCP_IMPORT_ERROR}",
        }

    try:
        mcp_tools = list_mcp_tools(server_config)
    except Exception as exc:
        return {
            "answer": "",
            "turns": 0,
            "tool_calls": [],
            "error": f"Could not list MCP tools: {exc}",
        }

    if not mcp_tools:
        return {
            "answer": "",
            "turns": 0,
            "tool_calls": [],
            "error": "The MCP server exposed no tools.",
        }

    openai_tools = mcp_tools_to_openai(mcp_tools)
    tool_lookup = {t.name: t for t in mcp_tools}

    messages: list = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with access to external tools via MCP. "
                "Use tools when they can answer the user's question more accurately. "
                "If a tool call fails, explain the error briefly and try another approach. "
                "Always ground your final answer in tool results."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    tool_calls_log: list = []

    for turn in range(1, max_turns + 1):
        try:
            resp = chat_client.chat.completions.create(
                model=model_id,
                messages=messages,
                tools=openai_tools,
                tool_choice="auto",
                temperature=temperature,
            )
        except Exception as exc:
            return {
                "answer": "",
                "turns": turn,
                "tool_calls": tool_calls_log,
                "error": f"LLM call failed: {exc}",
            }

        msg = resp.choices[0].message

        if not getattr(msg, "tool_calls", None):
            return {
                "answer": (msg.content or "").strip(),
                "turns": turn,
                "tool_calls": tool_calls_log,
                "error": None,
            }

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            if fn_name not in tool_lookup:
                result_text = f"Error: unknown tool '{fn_name}'."
                structured = None
                is_error = True
            else:
                try:
                    res = call_mcp_tool(server_config, fn_name, args)
                    result_text = res["text"] or json.dumps(
                        res.get("structured") or {}, ensure_ascii=False
                    )
                    structured = res.get("structured")
                    is_error = res["is_error"]
                except Exception as exc:
                    result_text = f"Tool execution error: {exc}"
                    structured = None
                    is_error = True

            tool_calls_log.append({
                "turn": turn,
                "tool": fn_name,
                "arguments": args,
                "is_error": is_error,
                "result_preview": result_text[:600],
                "structured": structured,
            })

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_text[:20_000],
            })

    return {
        "answer": "",
        "turns": max_turns,
        "tool_calls": tool_calls_log,
        "error": f"Reached max turns ({max_turns}) without a final answer.",
    }
