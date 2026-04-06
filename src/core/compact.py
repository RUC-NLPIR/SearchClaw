"""
Context compaction strategies.

When the conversation context grows too large, we compress it to stay
within the LLM's context window while preserving research findings.

Two-phase approach:
- Microcompact: full clear (not truncate) — partial data is worse than
  absent data, since the model may hallucinate missing portions.
- Full compact: 10,000 token output budget, research-specific prompt
  that emphasizes preserving source URLs, citations, and findings.
"""

from __future__ import annotations

import logging
import re

from src.core.types import Message
from src.utils.token_counter import estimate_tokens

logger = logging.getLogger(__name__)

# Cleared marker for old tool results
CLEARED_TOOL_RESULT = "[Old tool result content cleared]"

# How many recent messages to keep intact during microcompact.
MICROCOMPACT_KEEP_LAST_N = 5

# Minimum tool result size (chars) to bother clearing.
# Very small results are cheap to keep and may be useful.
MICROCOMPACT_MIN_SIZE = 500

# Max output tokens for the full compact summary.
# 10,000 tokens (~40K chars) is sufficient for a detailed
# research summary with all findings and source URLs.
FULL_COMPACT_MAX_TOKENS = 10000


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
    1. Microcompact: Clear old tool results entirely, keep last N intact
    2. Full compact: Summarize via side-query if still too large

    Returns a new message list (does not mutate the original).
    """
    compacted = list(messages)

    # --- Phase 1: Microcompact ---
    # Clear old tool results entirely, keeping only the last N
    compacted = _microcompact(compacted, keep_last_n=MICROCOMPACT_KEEP_LAST_N)

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
    keep_last_n: int = MICROCOMPACT_KEEP_LAST_N,
) -> list[Message]:
    """
    Clear old tool results entirely (not truncate).

    Tool results are either kept in full or replaced with a cleared
    marker. Truncating to N chars produces misleading partial data
    that can cause hallucination.

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

        # For older messages: clear large tool results entirely.
        # Assistant messages are left intact — they're typically short
        # ("Let me search for...") and the ones containing tool_use
        # blocks MUST be kept because the API requires tool_use_id
        # references to exist in a preceding assistant message.
        if msg.role == "tool":
            content = msg.text_content
            if len(content) > MICROCOMPACT_MIN_SIZE:
                result.append(Message(
                    role=msg.role,
                    content=CLEARED_TOOL_RESULT,
                    metadata=msg.metadata,
                ))
            else:
                result.append(msg)
        else:
            result.append(msg)

    return result


# ── Full compact prompt ─────────────────────────────────────────────
# Structured prompt tailored for a research agent that needs to
# preserve URLs, citations, and factual findings above all else.

_FULL_COMPACT_SYSTEM = (
    "You are summarizing a web research conversation. Your summary will "
    "replace the original conversation in the LLM's context, so it MUST "
    "preserve all important findings, source URLs, and citations. "
    "Respond with TEXT ONLY — do NOT call any tools."
)

_FULL_COMPACT_PROMPT = """\
Your task is to create a detailed summary of the research conversation so far. \
This summary replaces the full conversation, so it must be thorough in capturing \
all findings, sources, and context needed to continue the research.

Before providing your final summary, wrap your analysis in <analysis> tags to \
organize your thoughts and ensure completeness. Then provide the summary in \
<summary> tags.

Your summary MUST include the following sections:

1. **Research Query and Intent**: The original research question and what the \
user is trying to learn.

2. **Key Findings**: All factual findings discovered so far, organized by topic. \
Include specific data points, statistics, dates, and claims.

3. **Sources and Citations**: ALL source URLs mentioned in the conversation. \
These are CRITICAL — do not lose any URL. Format each as:
   - [Title or description](URL) — key finding from this source

4. **Tool Activity**: Which tools were used (web_search, web_fetch, \
academic_search, etc.) and what they found. Include search queries used.

5. **Contradictions and Gaps**: Any contradictions between sources, \
unanswered questions, or areas needing further research.

6. **All User Messages**: List ALL user messages (not tool results). \
These are critical for understanding evolving intent.

7. **Current State**: What was being researched immediately before this \
summary, and what the next logical research step would be.

Example structure:

<analysis>
[Your analysis ensuring all points are covered]
</analysis>

<summary>
1. Research Query and Intent:
   [Detailed description]

2. Key Findings:
   - [Finding 1 with specific details]
   - [Finding 2]

3. Sources and Citations:
   - [Source Title 1](https://example.com/1) — key finding from this source
   - [Source Title 2](https://example.com/2) — key finding from this source

4. Tool Activity:
   - web_search: [queries used and what was found]
   - web_fetch: [URLs fetched and key content]

5. Contradictions and Gaps:
   - [Any contradictions or open questions]

6. All User Messages:
   - [User message 1]
   - [User message 2]

7. Current State:
   [What was being worked on and next steps]
</summary>

Please provide your summary now.\
"""


async def _full_compact(messages: list[Message]) -> list[Message]:
    """
    Summarize the entire conversation via a side-query.

    Uses a cheap model to produce a comprehensive summary that preserves
    all research findings, source URLs, and citations.

    - No per-message cap — the summarizer needs full context
    - Structured prompt with <analysis> scratchpad (stripped from output)
    - 10,000 token output budget for detailed research summaries

    Returns a compacted message list: [original_query, summary, last_assistant].
    """
    from src.llm.client import side_query

    # Build a text representation of the conversation.
    # We do NOT cap each message — the summarizer needs full context
    # to produce an accurate summary.
    conv_parts = []
    for msg in messages:
        role = msg.role.upper()
        content = msg.text_content
        # Skip already-cleared tool results — they carry no information
        if content == CLEARED_TOOL_RESULT:
            conv_parts.append(f"[{role}]: {CLEARED_TOOL_RESULT}")
        else:
            conv_parts.append(f"[{role}]: {content}")
    conversation_text = "\n\n".join(conv_parts)

    # Summarize via side-query
    summary = await side_query(
        prompt=f"{_FULL_COMPACT_PROMPT}\n\n--- CONVERSATION ---\n{conversation_text}\n--- END ---",
        system=_FULL_COMPACT_SYSTEM,
        max_tokens=FULL_COMPACT_MAX_TOKENS,
    )

    if not summary:
        # Fallback: if side-query fails, just do aggressive microcompact
        logger.warning("Full compact side-query failed, falling back to aggressive microcompact")
        return _microcompact(messages, keep_last_n=2)

    # Strip <analysis> scratchpad from the summary. The analysis block
    # improves summary quality but has no value once the summary is written.
    summary = _format_compact_summary(summary)

    # Build compacted message list
    compacted = []

    # Keep the original user query
    for msg in messages:
        if msg.role == "user":
            compacted.append(msg)
            break

    # Add the summary as an assistant message
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


def _format_compact_summary(summary: str) -> str:
    """
    Strip <analysis> scratchpad and clean up <summary> tags.

    The <analysis> block is a drafting scratchpad that improves summary
    quality but has no informational value once the summary is written.
    """
    # Strip analysis section
    result = re.sub(r"<analysis>[\s\S]*?</analysis>", "", summary)

    # Extract summary section content
    match = re.search(r"<summary>([\s\S]*?)</summary>", result)
    if match:
        result = match.group(1).strip()

    # Clean up extra whitespace
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()
