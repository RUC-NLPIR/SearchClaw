"""
Content extraction — compresses large web pages before adding to context.

When a fetched web page exceeds a size threshold, a cheap side-query LLM
call extracts key facts, relevant URLs, and structural URLs (pagination,
sub-pages) instead of dumping the raw content into the main context.

This is a pre-injection compression step specific to the search agent.
Claude Code adds raw results and compacts later; for a research agent
doing synthesis across multiple pages, pre-extraction is more appropriate
because the model needs facts and attribution, not raw HTML artifacts.

The full raw content is always cached to disk for deep_read access.
"""

from __future__ import annotations

import logging

from src.llm.client import side_query

logger = logging.getLogger(__name__)

# Maximum chars of raw content to send to the extraction side-query.
# The extraction model has its own context limit — sending 200K chars
# of raw markdown would exceed it. We send a generous prefix that
# captures most article content.
_MAX_INPUT_FOR_EXTRACTION = 80000


async def extract_content(
    raw_content: str,
    research_query: str,
    source_url: str,
    source_title: str,
    max_output_chars: int = 5000,
) -> str | None:
    """
    Extract key facts and relevant URLs from raw web content.

    Uses a cheap side-query LLM to compress raw content into dense
    findings. Always preserves the source URL, structural URLs
    (pagination, sub-pages), and content URLs relevant to the query.
    Discards navigation, ads, social links, and unrelated content.

    Args:
        raw_content: Full markdown content of the web page.
        research_query: The user's research question (for relevance filtering).
        source_url: URL of the page (always preserved in output).
        source_title: Title of the page.
        max_output_chars: Rough target for output length.

    Returns:
        Extracted markdown content, or None if extraction fails
        (caller should fall back to raw truncation).
    """
    if not research_query:
        # Without a research question we can't filter for relevance —
        # return None so the caller falls back to raw truncation.
        logger.info("No research query available, skipping content extraction")
        return None

    # Truncate input to avoid exceeding the extraction model's context
    input_content = raw_content[:_MAX_INPUT_FOR_EXTRACTION]
    if len(raw_content) > _MAX_INPUT_FOR_EXTRACTION:
        input_content += (
            f"\n\n[... Content truncated for extraction. "
            f"Full content is {len(raw_content):,} chars.]"
        )

    prompt = _build_extraction_prompt(
        content=input_content,
        query=research_query,
        url=source_url,
        title=source_title,
        max_output_chars=max_output_chars,
    )

    try:
        result = await side_query(
            prompt=prompt,
            system=(
                "You are a research assistant extracting key information from a web page. "
                "Be thorough with facts and data but ruthlessly cut boilerplate and irrelevant content. "
                "Your output will be injected into a larger research context, so be concise and dense."
            ),
            max_tokens=2048,
        )

        if not result or len(result.strip()) < 50:
            logger.warning(
                f"Extraction returned insufficient content for {source_url}"
            )
            return None

        logger.info(
            f"Content extraction: {len(raw_content):,} chars → "
            f"{len(result):,} chars for {source_url}"
        )
        return result

    except Exception as e:
        logger.warning(f"Content extraction failed for {source_url}: {e}")
        return None


def _build_extraction_prompt(
    content: str,
    query: str,
    url: str,
    title: str,
    max_output_chars: int,
) -> str:
    """Build the extraction prompt for the side-query LLM."""
    return f"""Extract key information from this web page for a research task.

**Research question**: {query}
**Source URL**: {url}
**Page title**: {title}

Instructions — extract only what matters:

1. **KEY FACTS**: All factual claims, data points, statistics, dates, names, and findings relevant to the research question. Be specific — include numbers, percentages, dates, and proper nouns. Do not paraphrase vaguely; preserve precision.

2. **RELEVANT URLS**: Only URLs a researcher would want to follow up on:
   - References, cited studies, linked primary sources
   - Related articles directly relevant to the research question
   - PDF downloads, full reports, data appendices
   For each URL, include a brief note about what it links to.
   DO NOT include: social media share links, navigation menus, ad links, cookie/privacy policy links, "about us" or "contact" pages, login/signup links.

3. **STRUCTURAL URLS**: Always include these regardless of topic relevance:
   - Pagination links (next page, page 2, load more, etc.)
   - "Read more" / "Full article" / "Continue reading" links
   - Links to other sections or chapters of the same document
   - Download links (PDF, CSV, data files)
   Label these clearly as structural/navigation links.

4. **NOTABLE QUOTES**: 1-3 direct quotes worth citing, with attribution (author name, role if known).

Format as clean markdown. Start with:
## {{title}}
**Source**: {{url}}

Then present findings organized by topic. Keep URLs inline with context.

Target length: ~{max_output_chars:,} characters. Omit boilerplate, navigation text, ads, and content unrelated to the research question.

--- PAGE CONTENT ---
{content}
--- END ---"""
