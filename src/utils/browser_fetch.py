"""
Browser-based web page fetch — content extraction via Playwright.

Navigates to a URL in a browser, waits for the page to fully render
(including JS-driven content), and extracts the page content as clean
text.  Handles JS-rendered SPAs, auth-walled pages (when using the
user's real Chrome profile via CDP), and pages that block HTTP crawlers.

Used as a fallback in ``web_fetch.py`` between Jina Reader API and
direct ``httpx`` fetch when ``browser.use_for_fetch`` is enabled.

The extraction strategy:
1. Try common article selectors (article, main, .post-content, etc.)
2. If no article container found, strip boilerplate elements (nav,
   footer, sidebar, ads) and return the remaining body text
3. As a last resort, get the full HTML and convert via html_to_markdown()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.async_api import Page
    from src.utils.browser_manager import BrowserManager

logger = logging.getLogger(__name__)


async def browser_fetch(
    url: str,
    manager: BrowserManager = None,
    timeout: int = 30_000,
) -> dict | None:
    """
    Fetch a web page via browser — handles JS-rendered pages and auth-walled sites.

    Args:
        url: The URL to fetch.
        manager: BrowserManager instance for getting browser pages.
        timeout: Navigation timeout in milliseconds.

    Returns:
        {"title": str, "content": str, "url": str} on success,
        or None if fetch fails or returns insufficient content.
    """
    if manager is None:
        return None

    page = await manager.get_fetch_page()
    try:
        result = await _fetch_page_content(page, url, timeout)
        if result:
            logger.info(
                f"Browser fetch: url={url[:80]}, "
                f"title='{result['title'][:60]}', "
                f"content={len(result['content'])} chars"
            )
        return result
    except Exception as e:
        logger.warning(f"Browser fetch failed for {url}: {e}")
        return None
    finally:
        await manager.release_page(page, context_type="fetch")


async def _fetch_page_content(
    page: Page, url: str, timeout: int
) -> dict | None:
    """
    Navigate to URL and extract page content.

    Uses a multi-strategy approach:
    1. Navigate with networkidle wait (handles SPAs)
    2. Extract via article selectors → body text fallback → raw HTML
    """
    # Navigate to the page
    try:
        response = await page.goto(
            url, wait_until="networkidle", timeout=timeout
        )
    except Exception as e:
        # networkidle can be slow — retry with domcontentloaded
        logger.debug(
            f"networkidle timeout for {url}, retrying with domcontentloaded"
        )
        try:
            response = await page.goto(
                url, wait_until="domcontentloaded", timeout=timeout
            )
        except Exception as e2:
            logger.warning(f"Browser navigation failed for {url}: {e2}")
            return None

    if not response:
        return None

    # Check HTTP status
    status = response.status
    if status >= 400:
        logger.info(f"Browser fetch got HTTP {status} for {url}")
        return None

    # Get page title
    title = await page.title() or "Untitled Page"

    # Allow a brief moment for lazy-loaded content
    await page.wait_for_timeout(800)

    # --- Strategy 1: Extract from article/main container ---
    content = await page.evaluate("""() => {
        // Common article/content selectors, ordered by specificity
        const selectors = [
            'article',
            'main',
            '[role="main"]',
            '.post-content',
            '.entry-content',
            '.article-body',
            '.article-content',
            '.post-body',
            '.story-body',
            '#article-body',
            '.content-body',
            '.markdown-body',      // GitHub
            '.tweet-text',         // Twitter/X
            '.post',               // Generic blog posts
        ];

        for (const sel of selectors) {
            const el = document.querySelector(sel);
            if (el && el.innerText && el.innerText.trim().length > 200) {
                return el.innerText.trim();
            }
        }
        return null;
    }""")

    if content and len(content) >= 200:
        return {"title": title, "content": content, "url": url}

    # --- Strategy 2: Strip boilerplate, return remaining body text ---
    content = await page.evaluate("""() => {
        // Remove common boilerplate elements
        const removeSelectors = [
            'nav', 'footer', 'aside', 'header',
            '.sidebar', '.nav', '.navigation', '.menu',
            '.cookie-banner', '.cookie-notice', '.cookie-consent',
            '.ads', '.ad', '.advertisement', '[class*="advert"]',
            '.social-share', '.share-buttons',
            '.comments', '#comments', '.comment-section',
            '.related-posts', '.recommended',
            'script', 'style', 'noscript', 'iframe',
            '[role="banner"]', '[role="navigation"]',
            '[role="complementary"]', '[role="contentinfo"]',
        ];

        removeSelectors.forEach(sel => {
            document.querySelectorAll(sel).forEach(el => {
                try { el.remove(); } catch(e) {}
            });
        });

        const body = document.body;
        if (!body) return null;

        return body.innerText ? body.innerText.trim() : null;
    }""")

    if content and len(content) >= 50:
        return {"title": title, "content": content, "url": url}

    # --- Strategy 3: Get full HTML and convert via html_to_markdown ---
    try:
        html = await page.content()
        if html and len(html) > 100:
            from src.utils.html_to_markdown import html_to_markdown

            md_content = html_to_markdown(html, url=url)
            if md_content and len(md_content) >= 50:
                return {"title": title, "content": md_content, "url": url}
    except Exception as e:
        logger.debug(f"HTML-to-markdown fallback failed: {e}")

    logger.info(f"Browser fetch: insufficient content from {url}")
    return None
