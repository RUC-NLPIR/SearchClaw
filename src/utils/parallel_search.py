"""Anonymous Parallel Search MCP backend for the existing search_web tool."""

from __future__ import annotations

import asyncio
import hashlib
import json

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

MCP_URL = "https://search.parallel.ai/mcp"


async def search_parallel(
    client: httpx.AsyncClient,
    query: str,
    num_results: int,
    session_id: str,
    timeout: float,
) -> list[dict]:
    """Use Streamable HTTP, preserving timeout and task cancellation.

    Each call owns its MCP session; the HTTP client is shared and closed by
    Tool.aclose(). No API keys or saved credentials are read by this backend.
    """
    arguments = {"objective": query, "search_queries": [query]}
    if session_id:
        # Stable across turns, bounded to the server's 100-character limit.
        arguments["session_id"] = hashlib.sha256(session_id.encode()).hexdigest()

    async with asyncio.timeout(timeout):
        async with streamable_http_client(MCP_URL, http_client=client) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                response = await session.call_tool("web_search", arguments)

    if response.isError:
        raise ValueError("Parallel MCP returned a tool error")
    payload = response.structuredContent
    if payload is None:
        payload = json.loads(next(block.text for block in response.content if block.type == "text"))
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError("Parallel MCP returned an invalid search response")

    results = []
    for item in payload["results"][:num_results]:
        results.append({
            "title": item.get("title") or "Untitled",
            "link": item["url"],
            "snippet": "\n".join(item.get("excerpts") or []),
        })
    return results
