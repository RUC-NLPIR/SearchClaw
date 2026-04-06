"""
Browser-based web search — Google and DuckDuckGo search via Playwright.

Navigates to a search engine in a headless browser, waits for results
to render, and extracts structured results (title, URL, snippet) from
the DOM.  Returns the same format as Serper API so it can drop into
the existing fallback chain seamlessly.

IMPORTANT: Browser search only works reliably when connected to the
user's real Chrome via CDP mode. Search engines (Google, Bing, DDG)
aggressively detect automated browsers and show CAPTCHAs.  With the
user's real Chrome (real cookies, history, fingerprint), the browser
looks like a normal user.  Playwright's bundled Chromium with a fresh
profile is always blocked.

Used as a fallback in ``web_search.py`` when no ``SERPER_API_KEY``
is configured, ``browser.use_for_search`` is enabled, AND the browser
is in CDP mode.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from urllib.parse import quote_plus

if TYPE_CHECKING:
    from playwright.async_api import Page
    from src.utils.browser_manager import BrowserManager

logger = logging.getLogger(__name__)


async def browser_search(
    query: str,
    num_results: int = 10,
    search_engine: str = "google",
    manager: BrowserManager = None,
) -> list[dict]:
    """
    Search via browser — navigates to search engine, extracts results from DOM.

    Only works in CDP mode (user's real Chrome) because search engines
    block automated browsers with fresh profiles.  Returns empty list
    immediately if not in CDP mode.

    Returns same format as Serper API:
        [{"title": str, "link": str, "snippet": str}, ...]

    Args:
        query: The search query string.
        num_results: Maximum number of results to return.
        search_engine: "google" or "duckduckgo".
        manager: BrowserManager instance for getting browser pages.

    Returns:
        List of search result dicts, or empty list on failure.
    """
    if manager is None:
        return []

    # Browser search only works with the user's real Chrome (CDP mode).
    # Search engines aggressively block automated browsers with fresh
    # profiles — they always show CAPTCHAs.  With CDP, we inherit the
    # user's real cookies/history/fingerprint so it looks normal.
    if not manager.is_cdp_mode:
        logger.debug(
            "Browser search skipped — only works in CDP mode "
            "(user's real Chrome). Set browser.mode='cdp' and launch "
            "Chrome with --remote-debugging-port --user-data-dir to enable."
        )
        return []

    # In CDP mode, use the fetch page (which has the real Chrome context)
    page = await manager.get_fetch_page()
    try:
        if search_engine == "duckduckgo":
            results = await _search_duckduckgo(page, query, num_results)
        else:
            results = await _search_google(page, query, num_results)

        # Detect CAPTCHA/block even in CDP mode
        if not results:
            logger.info(
                f"Browser search ({search_engine}): no results for "
                f"'{query[:60]}' — search engine may have blocked the request"
            )
        else:
            logger.info(
                f"Browser search ({search_engine}): "
                f"query='{query[:60]}', results={len(results)}"
            )
        return results
    except Exception as e:
        logger.warning(f"Browser search failed: {e}")
        return []
    finally:
        await manager.release_page(page, context_type="fetch")


async def _search_google(
    page: Page, query: str, num_results: int
) -> list[dict]:
    """
    Google search via browser.

    1. Navigate to google.com/search?q={query}&num={num_results}
    2. Wait for results container to render
    3. Extract results via page.evaluate() — parse DOM for titles, URLs, snippets

    Google DOM selectors may change periodically.  The extraction JS
    tries multiple known selector patterns to be resilient.
    """
    url = (
        f"https://www.google.com/search?"
        f"q={quote_plus(query)}&num={min(num_results, 20)}&hl=en"
    )

    await page.goto(url, wait_until="domcontentloaded", timeout=15000)

    # Wait for search results to appear — try multiple selectors
    try:
        await page.wait_for_selector(
            "div#search, div#rso, div.g", timeout=10000
        )
    except Exception:
        logger.warning("Google search results container not found")
        return []

    # Allow a brief moment for dynamic content to finish rendering
    await page.wait_for_timeout(500)

    # Extract results from rendered DOM
    results = await page.evaluate(
        """(maxResults) => {
        const items = [];

        // Strategy 1: Standard Google result blocks (div.g)
        document.querySelectorAll('div.g').forEach(el => {
            if (items.length >= maxResults) return;

            const titleEl = el.querySelector('h3');
            const linkEl = el.querySelector('a[href^="http"]');
            // Snippets appear in various containers — try multiple selectors
            const snippetEl = el.querySelector(
                '[data-sncf], .VwiC3b, .IsZvec, .lEBKkf, ' +
                'div[style*="-webkit-line-clamp"], span.aCOpRe'
            );

            if (titleEl && linkEl) {
                const href = linkEl.href;
                // Skip Google's own links (images, maps, etc.)
                if (href.includes('google.com/search') ||
                    href.includes('google.com/maps') ||
                    href.includes('accounts.google.com')) {
                    return;
                }
                items.push({
                    title: titleEl.textContent.trim(),
                    link: href,
                    snippet: snippetEl
                        ? snippetEl.textContent.trim()
                        : ''
                });
            }
        });

        // Strategy 2: If div.g found nothing, try the rso container directly
        if (items.length === 0) {
            const rso = document.querySelector('#rso');
            if (rso) {
                rso.querySelectorAll('a[href^="http"]').forEach(linkEl => {
                    if (items.length >= maxResults) return;

                    // Find the closest heading
                    const heading = linkEl.querySelector('h3') ||
                                   linkEl.closest('[data-header-feature]')
                                         ?.querySelector('h3');
                    if (!heading) return;

                    const href = linkEl.href;
                    if (href.includes('google.com')) return;

                    // Avoid duplicates
                    if (items.some(i => i.link === href)) return;

                    // Find snippet — look in parent container
                    const parent = linkEl.closest('[data-hveid]') ||
                                   linkEl.parentElement?.parentElement;
                    const snippetTexts = parent
                        ? Array.from(parent.querySelectorAll('span, div'))
                              .filter(el => el.textContent.length > 40 &&
                                           el.textContent.length < 500 &&
                                           !el.querySelector('h3'))
                              .map(el => el.textContent.trim())
                        : [];

                    items.push({
                        title: heading.textContent.trim(),
                        link: href,
                        snippet: snippetTexts[0] || ''
                    });
                });
            }
        }

        return items;
    }""",
        num_results,
    )

    return results[:num_results]


async def _search_duckduckgo(
    page: Page, query: str, num_results: int
) -> list[dict]:
    """
    DuckDuckGo search via browser.

    DuckDuckGo's HTML is simpler and more stable than Google's.
    Results are in ``article[data-testid="result"]`` blocks.
    """
    url = f"https://duckduckgo.com/?q={quote_plus(query)}&ia=web"

    await page.goto(url, wait_until="domcontentloaded", timeout=15000)

    # Wait for results to render
    try:
        await page.wait_for_selector(
            "[data-testid='result'], .result, .results--main",
            timeout=10000,
        )
    except Exception:
        logger.warning("DuckDuckGo search results not found")
        return []

    await page.wait_for_timeout(500)

    results = await page.evaluate(
        """(maxResults) => {
        const items = [];

        // Strategy 1: Modern DDG (data-testid selectors)
        document.querySelectorAll('[data-testid="result"]').forEach(el => {
            if (items.length >= maxResults) return;

            const linkEl = el.querySelector('a[href^="http"]');
            const titleEl = el.querySelector('h2, [data-testid="result-title"]');
            const snippetEl = el.querySelector(
                '[data-testid="result-snippet"], .result__snippet'
            );

            if (linkEl && titleEl) {
                items.push({
                    title: titleEl.textContent.trim(),
                    link: linkEl.href,
                    snippet: snippetEl
                        ? snippetEl.textContent.trim()
                        : ''
                });
            }
        });

        // Strategy 2: Classic DDG HTML (fallback)
        if (items.length === 0) {
            document.querySelectorAll('.result, .web-result').forEach(el => {
                if (items.length >= maxResults) return;

                const linkEl = el.querySelector('a.result__a, a.result__url');
                const snippetEl = el.querySelector('.result__snippet');

                if (linkEl) {
                    items.push({
                        title: linkEl.textContent.trim(),
                        link: linkEl.href,
                        snippet: snippetEl
                            ? snippetEl.textContent.trim()
                            : ''
                    });
                }
            });
        }

        return items;
    }""",
        num_results,
    )

    return results[:num_results]
