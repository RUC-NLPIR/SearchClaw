"""
HTML to clean markdown conversion.

Uses trafilatura for content extraction (removes nav, ads, boilerplate)
and markdownify for HTML-to-markdown conversion. Falls back to a
simple tag-stripping approach if dependencies are unavailable.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


def html_to_markdown(html: str, url: str = "") -> str:
    """
    Convert HTML to clean, readable markdown.

    Strategy:
    1. Try trafilatura for content extraction (best at removing boilerplate)
    2. Fall back to markdownify (converts all HTML to markdown)
    3. Last resort: simple regex tag stripping

    Returns clean markdown text suitable for LLM consumption.
    """
    if not html:
        return ""

    # Strategy 1: trafilatura (best content extraction)
    try:
        import trafilatura
        result = trafilatura.extract(
            html,
            url=url,
            include_links=True,
            include_tables=True,
            include_images=False,
            include_comments=False,
            output_format="txt",
            favor_recall=True,
        )
        if result and len(result) > 100:
            return _clean_text(result)
    except Exception as e:
        logger.debug(f"trafilatura failed: {e}")

    # Strategy 2: markdownify
    try:
        from markdownify import markdownify as md
        result = md(
            html,
            heading_style="ATX",
            bullets="-",
            strip=["script", "style", "nav", "footer", "header", "aside"],
        )
        if result and len(result) > 50:
            return _clean_text(result)
    except Exception as e:
        logger.debug(f"markdownify failed: {e}")

    # Strategy 3: Simple tag stripping (last resort)
    return _strip_tags(html)


def _clean_text(text: str) -> str:
    """Clean up extracted text — normalize whitespace, remove artifacts."""
    # Collapse multiple blank lines to double
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove trailing whitespace from lines
    text = "\n".join(line.rstrip() for line in text.splitlines())
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def _strip_tags(html: str) -> str:
    """Simple HTML tag stripping as a last resort."""
    # Remove script and style content
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove HTML comments
    html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)
    # Remove tags
    html = re.sub(r"<[^>]+>", " ", html)
    # Decode common entities
    html = html.replace("&nbsp;", " ")
    html = html.replace("&amp;", "&")
    html = html.replace("&lt;", "<")
    html = html.replace("&gt;", ">")
    html = html.replace("&quot;", '"')
    html = html.replace("&#39;", "'")
    # Clean up whitespace
    return _clean_text(html)
