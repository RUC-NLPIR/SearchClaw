"""
ReAct baseline agent for benchmark comparison.

A minimal Reasoning + Acting loop with web_search and web_fetch tools.
No harness engineering: no quality hooks, no research plan, no content
extraction, no memory, no context compaction, no citation management.

Usage:
    from baseline.react.agent import react_agent
    result = await react_agent("What is X?", max_turns=30)
    print(result["answer"])
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import httpx
import litellm

logger = logging.getLogger(__name__)

# Suppress litellm noise
litellm.suppress_debug_info = True
litellm.drop_params = True

# Max chars of fetched content to inject into context (raw truncation)
MAX_CONTENT_CHARS = 20000


# ---------------------------------------------------------------------------
# Config loader — reuses the same settings.yaml as the main system
# ---------------------------------------------------------------------------

def load_config() -> dict:
    """Load LLM config from config/settings.yaml."""
    path = Path("config/settings.yaml")
    if not path.exists():
        return {}
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


_config = load_config()
_llm = _config.get("llm", {})
DEFAULT_MODEL = _llm.get("default_model", "anthropic/claude-sonnet-4-20250514")
BASE_URL = _llm.get("base_url", "") or ""
MAX_TOKENS = int(_llm.get("max_tokens", 4096))


# ---------------------------------------------------------------------------
# Tool definitions (for litellm function calling)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web and return a list of results with titles, URLs, and snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch a web page and return its content as text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch.",
                    },
                },
                "required": ["url"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations (standalone, no project dependencies)
# ---------------------------------------------------------------------------

async def web_search(query: str, num_results: int = 10) -> str:
    """Search via Serper API, fall back to DuckDuckGo HTML scraping."""
    api_key = os.environ.get("SERPER_API_KEY", "")

    async with httpx.AsyncClient(timeout=30) as client:
        # Try Serper first
        if api_key:
            try:
                resp = await client.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                    json={"q": query, "num": num_results},
                )
                resp.raise_for_status()
                data = resp.json()
                results = []
                for item in data.get("organic", []):
                    results.append(
                        f"- {item.get('title', '')}\n"
                        f"  URL: {item.get('link', '')}\n"
                        f"  Snippet: {item.get('snippet', '')}"
                    )
                if results:
                    return f"Search results for: {query}\n\n" + "\n\n".join(results)
            except Exception as e:
                logger.warning(f"Serper search failed: {e}")

        # Fallback: DuckDuckGo HTML scraping
        try:
            resp = await client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers={"User-Agent": "Mozilla/5.0 (compatible; ReActAgent/1.0)"},
                follow_redirects=True,
            )
            resp.raise_for_status()
            html = resp.text
            results = []
            blocks = re.findall(
                r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
                r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
                html, re.DOTALL,
            )
            for url, title, snippet in blocks[:num_results]:
                title = re.sub(r"<[^>]+>", "", title).strip()
                snippet = re.sub(r"<[^>]+>", "", snippet).strip()
                if "uddg=" in url:
                    import urllib.parse
                    parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
                    url = parsed.get("uddg", [url])[0]
                if url and title:
                    results.append(f"- {title}\n  URL: {url}\n  Snippet: {snippet}")
            if results:
                return f"Search results for: {query}\n\n" + "\n\n".join(results)
        except Exception as e:
            logger.warning(f"DuckDuckGo fallback failed: {e}")

    return "No search results found."


async def web_fetch(url: str) -> str:
    """Fetch via Jina Reader API, fall back to direct HTTP."""
    jina_key = os.environ.get("JINA_API_KEY", "")

    async with httpx.AsyncClient(timeout=60) as client:
        # Try Jina Reader first
        if jina_key:
            try:
                resp = await client.get(
                    f"https://r.jina.ai/{url}",
                    headers={
                        "Authorization": f"Bearer {jina_key}",
                        "Accept": "text/plain",
                        "X-Return-Format": "text",
                    },
                )
                resp.raise_for_status()
                content = resp.text
                if content.strip():
                    if len(content) > MAX_CONTENT_CHARS:
                        content = content[:MAX_CONTENT_CHARS] + "\n\n[Content truncated]"
                    return content
            except Exception as e:
                logger.warning(f"Jina fetch failed: {e}")

        # Fallback: direct HTTP
        try:
            resp = await client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; ReActAgent/1.0)"},
                follow_redirects=True,
            )
            resp.raise_for_status()
            text = resp.text
            # Strip HTML tags (basic)
            text = re.sub(r"<script[^>]*>[\s\S]*?</script>", "", text)
            text = re.sub(r"<style[^>]*>[\s\S]*?</style>", "", text)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) > MAX_CONTENT_CHARS:
                text = text[:MAX_CONTENT_CHARS] + "\n\n[Content truncated]"
            return text if text else "Failed to extract content from page."
        except Exception as e:
            logger.warning(f"Direct fetch failed: {e}")

    return f"Failed to fetch URL: {url}"


# Tool dispatch
TOOL_FUNCTIONS = {
    "web_search": web_search,
    "web_fetch": web_fetch,
}


# ---------------------------------------------------------------------------
# ReAct agent loop
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a research assistant. Answer the user's question by searching the web and reading pages.

You have access to two tools:
- web_search(query): Search the web for information.
- web_fetch(url): Fetch and read the content of a web page.

Strategy:
1. Think about what information you need.
2. Use web_search to find relevant pages.
3. Use web_fetch to read the most promising results.
4. Repeat as needed until you have enough information.
5. Provide your final answer.

Be thorough and verify information from multiple sources when possible.\
"""


async def react_agent(
    query: str,
    max_turns: int = 50,
    max_search: int = 20,
    max_fetch: int = 20,
    model: str | None = None,
) -> dict:
    """
    Run a ReAct agent loop on the given query.

    Returns:
        {
            "answer": str,
            "turn_count": int,
            "search_count": int,
            "fetch_count": int,
        }
    """
    target_model = model or DEFAULT_MODEL
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    search_count = 0
    fetch_count = 0
    rejected_turns = 0  # consecutive turns where ALL tool calls were rejected

    for turn in range(1, max_turns + 1):
        # Check per-tool limits — remove exhausted tools
        available_tools = []
        for t in TOOLS:
            name = t["function"]["name"]
            if name == "web_search" and search_count >= max_search:
                continue
            if name == "web_fetch" and fetch_count >= max_fetch:
                continue
            available_tools.append(t)

        # If all tools exhausted, force final answer (no more looping)
        if not available_tools:
            logger.info(f"All tool limits reached (search={search_count}, fetch={fetch_count}). Forcing final answer.")
            break

        # If the LLM keeps calling exhausted tools, stop the loop.
        # This handles models that ignore the `tools` parameter and
        # call tools they've seen in conversation history.
        if rejected_turns >= 3:
            logger.info(f"LLM called exhausted tools {rejected_turns} times in a row. Forcing final answer.")
            break

        # Call LLM
        kwargs = {
            "model": target_model,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "max_completion_tokens": MAX_TOKENS,
            "tools": available_tools,
        }
        if BASE_URL:
            kwargs["api_base"] = BASE_URL

        try:
            response = await litellm.acompletion(**kwargs)
        except Exception as e:
            logger.error(f"LLM call failed (turn {turn}): {e}")
            break

        choice = response.choices[0]
        message = choice.message

        # Append assistant message to history
        messages.append(message.model_dump())

        # If no tool calls, this is the final answer
        if not message.tool_calls:
            answer = message.content or ""
            return {
                "answer": answer,
                "turn_count": turn,
                "search_count": search_count,
                "fetch_count": fetch_count,
            }

        # Execute tool calls — enforce per-call limits strictly
        any_executed = False
        for tc in message.tool_calls:
            func_name = tc.function.name
            try:
                func_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                func_args = {}

            if func_name not in TOOL_FUNCTIONS:
                result = f"Unknown tool: {func_name}"
            elif func_name == "web_search" and search_count >= max_search:
                result = "Search limit reached. Please provide your final answer."
            elif func_name == "web_fetch" and fetch_count >= max_fetch:
                result = "Fetch limit reached. Please provide your final answer."
            else:
                any_executed = True
                # Increment count BEFORE execution so it's accurate
                if func_name == "web_search":
                    search_count += 1
                elif func_name == "web_fetch":
                    fetch_count += 1

                try:
                    result = await TOOL_FUNCTIONS[func_name](**func_args)
                except Exception as e:
                    result = f"Tool error: {e}"

            logger.info(
                f"Turn {turn}: {func_name}({json.dumps(func_args)[:80]}) "
                f"-> {len(result)} chars  [search={search_count}/{max_search}, fetch={fetch_count}/{max_fetch}]"
            )

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        # Track consecutive turns where the LLM only called exhausted tools
        if any_executed:
            rejected_turns = 0
        else:
            rejected_turns += 1

    # Max turns reached or all tools exhausted — force one last answer
    messages.append({
        "role": "user",
        "content": (
            "You have reached the maximum number of tool uses. "
            "Please provide your final answer now based on the information gathered so far."
        ),
    })

    # Anthropic requires `tools=` when conversation history contains
    # tool calls, so we must pass the full TOOLS list even here.
    kwargs = {
        "model": target_model,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "max_completion_tokens": MAX_TOKENS,
        "tools": TOOLS,
    }
    if BASE_URL:
        kwargs["api_base"] = BASE_URL

    try:
        response = await litellm.acompletion(**kwargs)
        msg = response.choices[0].message
        answer = msg.content or ""
        # If the LLM still returns tool calls instead of text, ignore them
        if not answer and msg.tool_calls:
            answer = ""
    except Exception as e:
        logger.error(f"Final answer failed: {e}")
        answer = ""

    return {
        "answer": answer,
        "turn_count": turn if 'turn' in dir() else max_turns,
        "search_count": search_count,
        "fetch_count": fetch_count,
    }
