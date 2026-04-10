"""
Relevance-based memory retrieval.

Mirrors Claude Code's findRelevantMemories.ts — scans memory
frontmatter headers and uses a side-query (cheap model) to select
the most relevant memories for the current research question.

Key design decisions (mirroring Claude Code):
- Always run the selector, even when memory count ≤ max_memories
- Use structured JSON output so the model can return an empty list
- System prompt explicitly allows "no relevant memories"
- On error: return [], never fall back to loading recent memories
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.memory.store import MemoryEntry, MemoryStore

logger = logging.getLogger(__name__)


async def find_relevant_memories(
    query: str,
    store: MemoryStore,
    max_memories: int = 5,
) -> list[MemoryEntry]:
    """
    Find memories relevant to the current query.

    Strategy:
    1. Load all memory headers (lightweight)
    2. Ask a cheap model to select the most relevant ones
    3. Load full content only for selected memories

    Mirrors Claude Code's findRelevantMemories() — always runs the
    selector (even for small memory counts) and allows empty results.
    On error, returns [] rather than injecting potentially irrelevant
    memories into the context.
    """
    headers = await store.get_headers()

    if not headers:
        return []

    # Always use the selector, even for small memory counts.
    # This prevents loading irrelevant memories into context.
    try:
        from src.llm.client import side_query

        headers_text = "\n".join(
            f"- {h['title']}: {h['preview']}"
            for h in headers
        )

        # Use structured JSON output so the model can return an empty list.
        # Mirrors Claude Code's SELECT_MEMORIES_SYSTEM_PROMPT which says:
        # "If there are no memories that would clearly be useful, feel free
        #  to return an empty list."
        response = await side_query(
            prompt=(
                f"Query: {query}\n\n"
                f"Available memories:\n{headers_text}"
            ),
            system=(
                "You are selecting memories that will be useful as context for "
                "processing a user's research query. Return a JSON object with a "
                "\"selected\" field containing a list of memory titles that are "
                "clearly relevant.\n\n"
                "Rules:\n"
                "- Only include memories you are CERTAIN will be helpful\n"
                "- If unsure, do NOT include it — be selective and discerning\n"
                "- If NO memories are relevant, return an empty list\n"
                f"- Maximum {max_memories} memories"
            ),
            max_tokens=256,
            output_schema={
                "type": "object",
                "properties": {
                    "selected": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Titles of relevant memories, or empty list if none are relevant",
                    }
                },
                "required": ["selected"],
                "additionalProperties": False,
            },
        )

        # Parse structured output
        try:
            parsed = json.loads(response)
            selected_titles = parsed.get("selected", [])
        except json.JSONDecodeError:
            # Fall back to text matching but still allow empty
            selected_titles = _parse_titles_from_text(response, headers)

        if not selected_titles:
            logger.info("Memory selector returned no relevant memories")
            return []

        # Match titles to headers and load content
        title_to_header = {h["title"]: h for h in headers}
        entries = []
        for title in selected_titles[:max_memories]:
            header = title_to_header.get(title)
            if header:
                path = Path(header["path"])
                entry = MemoryEntry.from_file(path)
                if entry:
                    entries.append(entry)

        if entries:
            logger.info(
                f"Memory selector picked {len(entries)} relevant memories: "
                + ", ".join(e.title for e in entries)
            )

        return entries

    except Exception as e:
        logger.warning(f"Memory selection failed: {e}")
        # On failure: return empty, NOT recent memories.
        # Mirrors Claude Code — never inject potentially irrelevant
        # content on error.
        return []


def _parse_titles_from_text(response: str, headers: list[dict]) -> list[str]:
    """Fallback: try to extract memory titles from unstructured text response."""
    titles = []
    response_lower = response.lower()
    for h in headers:
        if h["title"].lower() in response_lower:
            titles.append(h["title"])
    return titles


def format_memories_for_prompt(entries: list[MemoryEntry]) -> str:
    """
    Format memory entries for inclusion in the system prompt.

    Returns a concise text representation suitable for the LLM.
    """
    if not entries:
        return ""

    parts = []
    for entry in entries:
        parts.append(
            f"### [{entry.memory_type.value}] {entry.title}\n"
            f"{entry.content}\n"
        )

    return "\n".join(parts)
