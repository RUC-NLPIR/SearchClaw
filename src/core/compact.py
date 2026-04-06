"""
Context compaction strategies.

Follow Claude Code's auto-compact + microcompact patterns. When the
conversation context grows too large, we compress it to stay within
the LLM's context window while preserving research findings.

Strategies:
1. Microcompact: Strip raw tool results from early turns, keeping only
   summaries and citations.
2. Full compact: Summarize the entire conversation via a side-query.
"""

from __future__ import annotations

import logging

from src.core.types import Message
from src.utils.token_counter import estimate_tokens

logger = logging.getLogger(__name__)


def should_compact(
    messages: list[Message],
    threshold_tokens: int = 80000,
) -> bool:
    """
    Check if the conversation is large enough to need compaction.

    Estimates token count across all messages and compares to threshold.
    """
    total = sum(estimate_tokens(msg.text_content) for msg in messages)
    if total > threshold_tokens:
        logger.info(f"Context at ~{total:,} tokens, threshold is {threshold_tokens:,} — compaction needed")
        return True
    return False


async def compact_messages(
    messages: list[Message],
    threshold_tokens: int = 80000,
) -> list[Message]:
    """
    Compact the conversation to reduce context size.

    Two-phase approach:
    1. Microcompact: Strip large tool results from older turns
    2. Full compact: Summarize via side-query if still too large

    Returns a new message list (does not mutate the original).
    """
    compacted = list(messages)

    # --- Phase 1: Microcompact ---
    # Strip raw tool results from older turns, keeping only the last N
    compacted = _microcompact(compacted, keep_last_n=4)

    # Check if we're now under the threshold
    total = sum(estimate_tokens(msg.text_content) for msg in compacted)
    if total <= threshold_tokens:
        logger.info(f"Microcompact sufficient: ~{total:,} tokens")
        return compacted

    # --- Phase 2: Full compact via side-query ---
    logger.info(f"Microcompact not enough (~{total:,} tokens), doing full compact")
    compacted = await _full_compact(compacted)

    return compacted


def _microcompact(
    messages: list[Message],
    keep_last_n: int = 4,
) -> list[Message]:
    """
    Strip large tool results from older messages.

    Older tool results are replaced with a brief summary, preserving 
    only the most recent results in full.
    Keeps the first user message and last N messages intact.
    """
    if len(messages) <= keep_last_n + 1:
        return messages

    # Preserve: first user message + last N messages
    result = []
    cutoff = len(messages) - keep_last_n

    for i, msg in enumerate(messages):
        if i == 0:
            # Always keep the original query
            result.append(msg)
            continue

        if i >= cutoff:
            # Keep recent messages intact
            result.append(msg)
            continue

        # For older messages: strip large tool results
        if msg.role == "tool":
            content = msg.text_content
            if len(content) > 500:
                # Replace with truncated version
                truncated = content[:300] + "\n\n[... content compacted to save context ...]"
                result.append(Message(
                    role=msg.role,
                    content=truncated,
                    metadata=msg.metadata,
                ))
            else:
                result.append(msg)
        elif msg.role == "assistant" and isinstance(msg.content, str) and len(msg.content) > 1000:
            # Truncate long assistant text from early turns
            truncated = msg.content[:500] + "\n\n[... earlier reasoning compacted ...]"
            result.append(Message(
                role=msg.role,
                content=truncated,
                metadata=msg.metadata,
            ))
        else:
            result.append(msg)

    return result


async def _full_compact(messages: list[Message]) -> list[Message]:
    """
    Summarize the entire conversation via a side-query.

    Uses a cheap model to produce a comprehensive summary that preserves:
    - The original query
    - All key findings and facts discovered
    - All source URLs and citations
    - Any unresolved questions or leads

    Returns a compacted message list: [original_query, system_summary, last_assistant_message].
    """
    from src.llm.client import side_query

    # Build a text representation of the conversation
    conv_parts = []
    for msg in messages:
        role = msg.role.upper()
        content = msg.text_content[:2000]  # Cap each message for the summary prompt
        conv_parts.append(f"[{role}]: {content}")
    conversation_text = "\n\n".join(conv_parts)

    # Summarize via side-query (cheap model)
    summary = await side_query(
        prompt=(
            "Summarize the following research conversation. Preserve ALL of these:\n"
            "1. The original research question\n"
            "2. All factual findings discovered so far\n"
            "3. All source URLs mentioned (these are critical — don't lose any)\n"
            "4. Any unresolved questions or leads to follow up on\n"
            "5. Any contradictions or disagreements between sources\n\n"
            "Be comprehensive but concise. This summary replaces the full conversation.\n\n"
            f"--- CONVERSATION ---\n{conversation_text}\n--- END ---"
        ),
        system=(
            "You are summarizing a web research conversation. Your summary will replace "
            "the original conversation in the LLM's context, so it MUST preserve all "
            "important findings and source URLs. Be thorough but concise."
        ),
        max_tokens=1024,
    )

    if not summary:
        # Fallback: if side-query fails, just do aggressive microcompact
        logger.warning("Full compact side-query failed, falling back to aggressive microcompact")
        return _microcompact(messages, keep_last_n=2)

    # Build compacted message list
    compacted = []

    # Keep the original user query
    for msg in messages:
        if msg.role == "user":
            compacted.append(msg)
            break

    # Add the summary as a system message
    compacted.append(Message(
        role="assistant",
        content=(
            "[Previous research has been summarized to save context space]\n\n"
            f"{summary}"
        ),
        metadata={"compacted": True},
    ))

    # Keep the last assistant message if it exists and is different
    for msg in reversed(messages):
        if msg.role == "assistant" and not msg.metadata.get("compacted"):
            if msg.text_content != compacted[-1].text_content:
                compacted.append(msg)
            break

    return compacted
