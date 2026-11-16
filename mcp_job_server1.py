# job_hunt_agent_groq.py
"""
job_hunt_agent_groq.py — Groq-powered agent for mcp_job_server.py.

Uses openai/gpt-oss-120b on Groq. Supports interactive chat and one-shot CLI queries.

Usage:
    export GROQ_API_KEY="gsk_..."
    python job_hunt_agent_groq.py mcp_job_server.py
    python job_hunt_agent_groq.py mcp_job_server.py --query "Find me data engineer roles in Singapore and scrape the top one"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from typing import Any

from groq import Groq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

MODEL = "openai/gpt-oss-120b"
MAX_HISTORY_MESSAGES = 30


class JobHuntAgentGroq:
    def __init__(self, server_script: str) -> None:
        if not os.environ.get("GROQ_API_KEY"):
            raise SystemExit("Set GROQ_API_KEY before running.")
        self.server_script = server_script
        self.client = Groq()
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self.groq_tools: list[dict[str, Any]] = []

        self.system_prompt = (
            "You are an AI job-hunting copilot. You help the user discover, "
            "extract, and organize job postings.\n\n"
            "Guidelines:\n"
            "- Use `search_jobs` to find candidate postings.\n"
            "- Use `scrape_job_url` to pull company, role, and JD from a specific URL.\n"
            "- After scraping, summarize the role in 3 bullets and flag any skill "
            "gaps versus the user's stated background.\n"
            "- Never invent company names, salaries, or URLs.\n"
            "- When scraping fails (Cloudflare / JS-only pages), report the error "
            "honestly and suggest the user paste the JD manually.\n"
            "- Keep responses tight; bullets beat paragraphs."
        )

    # ---------- lifecycle ----------
    async def connect(self) -> None:
        params = StdioServerParameters(
            command=sys.executable,
            args=[self.server_script],
            env=None,
        )
        read, write = await self.exit_stack.enter_async_context(stdio_client(params))
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await self.session.initialize()

        mcp_tools = (await self.session.list_tools()).tools
        self.groq_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema,
                },
            }
            for t in mcp_tools
        ]
        print(f"[agent] connected — tools: {[t['function']['name'] for t in self.groq_tools]}")

    async def close(self) -> None:
        await self.exit_stack.aclose()

    # ---------- core loop ----------
    async def ask(self, query: str) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]

        while True:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=self.groq_tools,
                tool_choice="auto",
                temperature=0.3,
                max_completion_tokens=2048,
            )
            msg = response.choices[0].message
            tool_calls = msg.tool_calls or []

            # If no tools were called, we're done
            if not tool_calls:
                # Groq sometimes puts reasoning in `reasoning` field; we ignore it for brevity
                return (msg.content or "").strip()

            # Append the assistant message (with tool_calls) to history
            messages.append(msg)

            # Execute each tool call
            for tool_call in tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                print(f"[tool] {fn_name}({fn_args})")

                try:
                    result = await self.session.call_tool(fn_name, fn_args)
                    payload = (
                        result.content[0].text
                        if result.content
                        else "{}"
                    )
                except Exception as exc:
                    payload = f"tool error: {exc}"

                messages.append({
                    "role": "tool",
                    "content": payload,
                    "tool_calls_id": tool_call.id,  # Groq expects this key
                })

            # Trim old history if it gets too long
            if len(messages) > MAX_HISTORY_MESSAGES:
                # Keep the system prompt and the most recent exchanges
                messages = [messages[0]] + messages[-(MAX_HISTORY_MESSAGES - 1):]

    # ---------- interfaces ----------
    async def chat_loop(self) -> None:
        print("\nJob Hunt Agent (Groq) — type 'quit' to exit.\n")
        while True:
            try:
                q = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q or q.lower() in {"quit", "exit"}:
                break
            try:
                print(f"\nAgent: {await self.ask(q)}\n")
            except Exception as exc:
                print(f"\n[error] {exc}\n")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("server", help="Path to mcp_job_server.py")
    parser.add_argument("--query", help="Run a single query and exit")
    args = parser.parse_args()

    agent = JobHuntAgentGroq(args.server)
    await agent.connect()
    try:
        if args.query:
            print(await agent.ask(args.query))
        else:
            await agent.chat_loop()
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
