"""
Exa search tool — semantic web search via Exa's MCP server.

Uses the official Exa MCP endpoint (https://mcp.exa.ai/mcp) via
Streamable HTTP transport (JSON-RPC over HTTP). This avoids the need
for a separate MCP client library — only httpx is required.

Authentication:
  - If EXA_API_KEY is set, it is appended to the MCP URL for higher
    rate limits and priority access.
  - Without EXA_API_KEY, the MCP endpoint is called directly.
    This still works for basic searches (rate-limited).

Concurrency-safe: multiple Exa searches can run in parallel.
"""

from __future__ import annotations

import logging
import os
import uuid

import httpx

from src.core.tool import Tool, ToolUseContext
from src.core.types import Citation, SourceType, ToolResult, ValidationResult

logger = logging.getLogger(__name__)

# Exa hosted MCP endpoint (Streamable HTTP transport)
EXA_MCP_URL = "https://mcp.exa.ai/mcp"


class ExaSearchTool(Tool):
    name = "exa_search"
    description = (
        "Search the web using Exa's semantic search engine. "
        "Returns results with titles, URLs, and content highlights. "
        "Good for finding specific topics, research, and high-quality sources. "
        "Complements web_search with different ranking and coverage."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query. Exa works well with natural language questions.",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (default: 10, max: 20).",
                "default": 10,
            },
            "category": {
                "type": "string",
                "description": (
                    "Optional: focus on a specific category. "
                    "Options: 'research paper', 'news', 'company', 'personal site'."
                ),
                "enum": ["research paper", "news", "company", "personal site"],
            },
        },
        "required": ["query"],
    }

    is_concurrency_safe = True
    is_read_only = True

    def __init__(
        self,
        default_results: int = 10,
        max_results: int = 20,
        max_result_size_chars: int = 15000,
        http_timeout: int = 30,
    ):
        self.default_results = default_results
        self.max_results = max_results
        self.max_result_size_chars = max_result_size_chars
        self._client = httpx.AsyncClient(timeout=float(http_timeout))
        self._mcp_session_id: str | None = None

    def prompt(self) -> str:
        return (
            "Use exa_search for semantic web search — it understands natural language "
            "queries and finds high-quality, relevant results. Tips:\n"
            "- Use natural language questions (e.g. 'what are the latest advances in RAG?')\n"
            "- Use category='research paper' for academic content\n"
            "- Use category='news' for recent news articles\n"
            "- Complements web_search — use both for comprehensive coverage\n"
            "- After searching, use web_fetch to read the most promising results"
        )

    def validate_input(self, args: dict) -> ValidationResult:
        query = args.get("query", "")
        if not query or len(query.strip()) < 2:
            return ValidationResult(valid=False, message="Query must be at least 2 characters")
        if len(query) > 500:
            return ValidationResult(valid=False, message="Query too long (max 500 chars)")
        return ValidationResult(valid=True)

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        query = args["query"]
        num_results = min(args.get("num_results", self.default_results), self.max_results)
        category = args.get("category")

        results = await self._search_via_mcp(query, num_results, category)

        if not results:
            return ToolResult(
                data="No Exa search results found. Try a different query or use web_search.",
                is_error=False,
            )

        # Format results
        formatted_parts = [f"## Exa Search Results for: {query}\n"]
        citations = []

        for i, r in enumerate(results, 1):
            title = r.get("title", "Untitled")
            url = r.get("url", "")
            snippet = r.get("snippet", "No description available")

            formatted_parts.append(
                f"### {i}. {title}\n"
                f"**URL**: {url}\n"
                f"**Snippet**: {snippet}\n"
            )

            citations.append(Citation(
                url=url,
                title=title,
                snippet=snippet,
                source_type=SourceType.WEB,
            ))

        formatted = "\n".join(formatted_parts)
        formatted, truncated, cached_path = await self._maybe_truncate(
            formatted, query, context
        )

        return ToolResult(
            data=formatted,
            citations=citations,
            truncated=truncated,
            cached_path=cached_path,
        )

    # ------------------------------------------------------------------
    # MCP Streamable HTTP transport
    # ------------------------------------------------------------------

    def _build_mcp_url(self) -> str:
        """Build the MCP endpoint URL, appending API key if available."""
        exa_key = os.environ.get("EXA_API_KEY", "")
        if exa_key:
            return f"{EXA_MCP_URL}?exaApiKey={exa_key}"
        return EXA_MCP_URL

    async def _mcp_initialize(self) -> str | None:
        """
        Send MCP initialize request.

        Returns the session ID on success, or None on failure.
        The session ID (Mcp-Session-Id header) must be included
        in subsequent requests.
        """
        request = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {
                    "name": "searchclaw",
                    "version": "0.1.0",
                },
            },
        }

        try:
            response = await self._client.post(
                self._build_mcp_url(),
                json=request,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
            )
            response.raise_for_status()

            # Extract session ID from response header
            session_id = response.headers.get("mcp-session-id")
            self._mcp_session_id = session_id

            # Send initialized notification
            notif = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            }
            headers = {"Content-Type": "application/json",
                        "Accept": "application/json, text/event-stream"}
            if session_id:
                headers["Mcp-Session-Id"] = session_id
            await self._client.post(
                self._build_mcp_url(),
                json=notif,
                headers=headers,
            )

            return session_id
        except Exception as e:
            logger.warning(f"Exa MCP initialize failed: {e}")
            return None

    async def _search_via_mcp(
        self,
        query: str,
        num_results: int,
        category: str | None = None,
    ) -> list[dict]:
        """
        Search via Exa's hosted MCP server using Streamable HTTP transport.

        Sends a JSON-RPC tools/call request to invoke the web_search_exa tool.
        """
        # Initialize MCP session if needed
        if not self._mcp_session_id:
            session_id = await self._mcp_initialize()
            if session_id is None:
                logger.warning("Exa MCP initialization failed, trying direct call")

        # Build tool arguments
        tool_args: dict = {
            "query": query,
            "numResults": num_results,
        }
        if category:
            tool_args["category"] = category

        request = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {
                "name": "web_search_exa",
                "arguments": tool_args,
            },
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._mcp_session_id:
            headers["Mcp-Session-Id"] = self._mcp_session_id

        try:
            response = await self._client.post(
                self._build_mcp_url(),
                json=request,
                headers=headers,
            )
            response.raise_for_status()
        except httpx.TimeoutException:
            logger.warning("Exa MCP request timed out")
            return []
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                logger.warning(
                    "Exa MCP auth required. Set EXA_API_KEY for authenticated access."
                )
            else:
                logger.warning(f"Exa MCP HTTP error {e.response.status_code}")
            # Reset session for next attempt
            self._mcp_session_id = None
            return []
        except Exception as e:
            logger.warning(f"Exa MCP request failed: {e}")
            self._mcp_session_id = None
            return []

        # Parse JSON-RPC response
        try:
            rpc_response = response.json()
        except Exception:
            logger.warning("Exa MCP returned non-JSON response")
            return []

        if "error" in rpc_response:
            err = rpc_response["error"]
            logger.warning(f"Exa MCP error: {err.get('message', err)}")
            return []

        result = rpc_response.get("result", {})
        return self._parse_mcp_result(result)

    def _parse_mcp_result(self, result: dict) -> list[dict]:
        """
        Parse the MCP tools/call result into a list of search results.

        The MCP tool result has a "content" field with an array of content
        blocks. The actual search data is typically in a text block as
        formatted text or JSON.
        """
        content_blocks = result.get("content", [])
        results = []

        for block in content_blocks:
            if block.get("type") != "text":
                continue

            text = block.get("text", "")

            # Try to parse as JSON (Exa MCP may return structured results)
            try:
                import json
                data = json.loads(text)
                # Handle case where data is a list of results
                if isinstance(data, list):
                    for item in data:
                        results.append({
                            "title": item.get("title", "Untitled"),
                            "url": item.get("url", ""),
                            "snippet": (
                                item.get("summary", "")
                                or item.get("text", "")[:300]
                                or item.get("highlights", [""])[0]
                                or "No description"
                            ),
                        })
                # Handle case where data has a "results" key
                elif isinstance(data, dict) and "results" in data:
                    for item in data["results"]:
                        results.append({
                            "title": item.get("title", "Untitled"),
                            "url": item.get("url", ""),
                            "snippet": (
                                item.get("summary", "")
                                or item.get("text", "")[:300]
                                or item.get("highlights", [""])[0]
                                or "No description"
                            ),
                        })
                continue
            except (json.JSONDecodeError, ValueError):
                pass

            # Fall back to text parsing if not JSON
            # Exa MCP may return formatted text like:
            #   Title: ...
            #   URL: ...
            #   Summary: ...
            results.extend(self._parse_text_results(text))

        return results

    def _parse_text_results(self, text: str) -> list[dict]:
        """Parse text-formatted search results from Exa MCP."""
        import re
        results = []
        # Try to split by numbered items or separator lines
        blocks = re.split(r'\n(?=\d+\.\s|\-{3,}|Title:)', text)

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            title_match = re.search(r'(?:Title:|^\d+\.\s*)(.*?)(?:\n|$)', block)
            url_match = re.search(r'(?:URL:|Source:|Link:)\s*(https?://\S+)', block)

            if title_match and url_match:
                title = title_match.group(1).strip()
                url = url_match.group(1).strip()
                # Extract whatever text remains as snippet
                snippet = re.sub(
                    r'(?:Title:|URL:|Source:|Link:|^\d+\.).*\n?', '', block
                ).strip()[:300] or "No description"
                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                })

        return results
