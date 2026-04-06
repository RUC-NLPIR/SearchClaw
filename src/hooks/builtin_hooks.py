"""
Built-in quality hooks.

These hooks implement common quality checks that run before the agent
finalizes its answer. They can force additional research if the answer
doesn't meet quality standards.

Non-research queries (greetings, identity questions, simple chat) are
detected and exempt from citation/completeness requirements. Detection
uses a cheap LLM side-query for generalizability rather than brittle
regex patterns.
"""

from __future__ import annotations

import json
import logging

from src.core.types import LoopState
from src.hooks.engine import Hook, HookEvaluation

logger = logging.getLogger(__name__)

# Cache for _needs_research() — avoids redundant side-query calls
# when multiple hooks check the same state in a single stop-hook cycle.
# Uses a WeakValueDictionary-like approach: keyed by id(state) but
# bounded to prevent unbounded growth. Cleared after each stop-hook
# evaluation cycle via clear_research_cache().
_research_cache: dict[int, bool] = {}
_RESEARCH_CACHE_MAX_SIZE = 50


def clear_research_cache() -> None:
    """Clear the research classification cache between evaluation cycles."""
    _research_cache.clear()


async def _needs_research(state: LoopState) -> bool:
    """
    Determine whether the current query requires research with citations.

    Uses a cheap LLM side-query to classify the query intent. Returns
    False for conversational queries (greetings, identity questions,
    chitchat) that don't need citations or source verification.

    Key signal: if the model already used tools, the query clearly needed
    research — skip the classification call entirely.
    """
    # Check cache first — multiple hooks call this with the same state
    cache_key = id(state)
    if cache_key in _research_cache:
        return _research_cache[cache_key]

    # Prevent unbounded growth — evict all if too large
    if len(_research_cache) >= _RESEARCH_CACHE_MAX_SIZE:
        _research_cache.clear()

    # If any tool was used, the model decided research was needed
    has_tool_use = any(m.role == "tool" for m in state.messages)
    if has_tool_use:
        _research_cache[cache_key] = True
        return True

    # Find the first user message (the original query)
    first_user_msg = None
    for msg in state.messages:
        if msg.role == "user":
            first_user_msg = msg.text_content.strip()
            break

    if not first_user_msg:
        _research_cache[cache_key] = True
        return True  # No query found — default to requiring research

    # Use a side-query to classify the query intent
    try:
        from src.llm.client import side_query

        response = await side_query(
            prompt=f"User query: \"{first_user_msg}\"",
            system=(
                "Classify whether this user query requires web research and citations "
                "to answer properly. Return a JSON object with a single boolean field.\n\n"
                "Return {\"needs_research\": false} for:\n"
                "- Greetings, farewells, thanks\n"
                "- Identity questions about the assistant itself\n"
                "- Simple chitchat or conversational remarks\n"
                "- Requests that are about the assistant's capabilities\n"
                "- Any query the assistant can answer from its own nature\n\n"
                "Return {\"needs_research\": true} for:\n"
                "- Factual questions about the world\n"
                "- Questions requiring up-to-date information\n"
                "- Technical, scientific, or domain-specific questions\n"
                "- Any query where citing sources adds value"
            ),
            max_tokens=32,
            output_schema={
                "type": "object",
                "properties": {
                    "needs_research": {
                        "type": "boolean",
                        "description": "Whether the query requires research and citations",
                    }
                },
                "required": ["needs_research"],
                "additionalProperties": False,
            },
        )

        parsed = json.loads(response)
        result = parsed.get("needs_research", True)
        if not result:
            logger.info(f"Query classified as non-research: \"{first_user_msg[:80]}\"")
        _research_cache[cache_key] = result
        return result

    except Exception as e:
        logger.warning(f"Research classification failed: {e}, defaulting to True")
        _research_cache[cache_key] = True
        return True  # On error, assume research is needed (fail-safe)


class CitationQualityHook(Hook):
    """
    Checks that the answer has sufficient citations.

    If the answer cites fewer than min_citations sources, it forces
    the agent to search for more evidence.

    Memory-aware: when the agent answers without any tool use (e.g.
    drawing on memories injected in the system prompt), the feedback
    is targeted — "verify and cite your claims" rather than a generic
    "search more". This avoids a wasted turn where the agent blindly
    re-researches topics it already understands from memory.
    """
    name = "citation_quality"
    description = "Ensures the answer has enough citations"

    def __init__(self, min_citations: int = 2):
        self.min_citations = min_citations

    async def evaluate(self, state: LoopState, **kwargs) -> HookEvaluation:
        if self.min_citations == 0:
            return HookEvaluation(passed=True)

        # Skip citation requirements for non-research queries
        if not await _needs_research(state):
            return HookEvaluation(passed=True)

        num_citations = len(state.citations)

        if num_citations >= self.min_citations:
            return HookEvaluation(passed=True)

        # Check whether ANY tool was used in this session so far
        has_tool_use = any(m.role == "tool" for m in state.messages)

        if not has_tool_use:
            # Agent answered without calling any tools — likely drew on
            # memory content or prior knowledge. Give targeted feedback
            # so the next iteration verifies rather than re-researches.
            return HookEvaluation(
                passed=False,
                feedback=(
                    "You answered without searching for sources. Even if you have "
                    "prior knowledge or memory context about this topic, you MUST "
                    "verify your claims with current sources.\n\n"
                    "Please:\n"
                    "1. Search for sources that confirm the key facts in your answer\n"
                    "2. Use cite_source to register each source\n"
                    "3. Then provide your final answer with inline citations"
                ),
            )

        # Agent used tools but still has too few citations
        return HookEvaluation(
            passed=False,
            feedback=(
                f"Your answer only cites {num_citations} source(s), but at least "
                f"{self.min_citations} are needed. Please search for additional sources "
                f"to support your claims, then cite them using the cite_source tool."
            ),
        )


class SourceDiversityHook(Hook):
    """
    Ensures the answer doesn't rely on a single source.

    If all citations come from the same domain, it forces the agent
    to find corroborating sources from different domains.
    """
    name = "source_diversity"
    description = "Ensures citations come from diverse sources"

    def __init__(self, min_domains: int = 2):
        self.min_domains = min_domains

    async def evaluate(self, state: LoopState, **kwargs) -> HookEvaluation:
        # Skip for non-research queries
        if not await _needs_research(state):
            return HookEvaluation(passed=True)

        if len(state.citations) < 2:
            # Not enough citations to check diversity — let citation hook handle it
            return HookEvaluation(passed=True)

        # Extract unique domains
        from urllib.parse import urlparse
        domains = set()
        for citation in state.citations:
            try:
                domain = urlparse(citation.url).netloc
                # Normalize: remove www.
                domain = domain.replace("www.", "")
                domains.add(domain)
            except Exception:
                continue

        if len(domains) < self.min_domains:
            return HookEvaluation(
                passed=False,
                feedback=(
                    f"All your citations come from {len(domains)} domain(s): "
                    f"{', '.join(domains)}. Please search for corroborating information "
                    f"from at least {self.min_domains} different sources/domains."
                ),
            )

        return HookEvaluation(passed=True)


class AnswerCompletenessHook(Hook):
    """
    Basic check that the answer has substantive content.

    If the assistant's last message is too short, it's likely incomplete
    or a cop-out ("I couldn't find information").
    """
    name = "answer_completeness"
    description = "Ensures the answer is substantive"

    def __init__(self, min_chars: int = 200):
        self.min_chars = min_chars

    async def evaluate(self, state: LoopState, **kwargs) -> HookEvaluation:
        # Skip length requirements for non-research queries
        if not await _needs_research(state):
            return HookEvaluation(passed=True)

        last_msg = state.last_assistant_message

        if not last_msg:
            return HookEvaluation(
                passed=False,
                feedback="You haven't provided an answer yet. Please synthesize your research findings.",
            )

        if len(last_msg) < self.min_chars:
            return HookEvaluation(
                passed=False,
                feedback=(
                    f"Your answer is only {len(last_msg)} characters, which seems too brief. "
                    f"Please provide a more comprehensive response with supporting details "
                    f"from your research."
                ),
            )

        return HookEvaluation(passed=True)


def build_default_hooks(config: dict | None = None) -> list[Hook]:
    """
    Create the default set of quality hooks.

    Args:
        config: Optional dict from settings.yaml hooks section.
                Keys: "min_citations", "min_domains", "min_answer_chars".

    Returns a list of hooks that should be registered as stop hooks.
    """
    cfg = config or {}
    return [
        CitationQualityHook(min_citations=cfg.get("min_citations", 2)),
        SourceDiversityHook(min_domains=cfg.get("min_domains", 2)),
        AnswerCompletenessHook(min_chars=cfg.get("min_answer_chars", 200)),
    ]
