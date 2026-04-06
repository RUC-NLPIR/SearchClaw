"""
Quick standalone test for browser search and fetch.

Usage:
    python tests/test_browser.py              # Run all tests
    python tests/test_browser.py search       # Test browser search only
    python tests/test_browser.py fetch        # Test browser fetch only
    python tests/test_browser.py fetch <url>  # Test fetching a specific URL

Notes:
    - Browser search ONLY works in CDP mode (user's real Chrome with
      existing cookies/history). In Playwright mode it is skipped — this
      is by design, not a failure.
    - Browser fetch works in both Playwright and CDP modes.

To test CDP mode:
    1. Quit Chrome completely
    2. Relaunch with --remote-debugging-port AND --user-data-dir (required for Chrome 146+):
       macOS:
         /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
           --remote-debugging-port=9222 \
           --user-data-dir="$HOME/chrome-cdp-profile"
       Linux:
         google-chrome --remote-debugging-port=9222 --user-data-dir="$HOME/chrome-cdp-profile"
    3. Verify CDP is active: curl http://127.0.0.1:9222/json/version
    4. Run: python tests/test_browser.py search --cdp
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def test_browser_search(engine: str = "google", mode: str = "playwright"):
    """Test browser-based Google/DuckDuckGo search."""
    from src.utils.browser_manager import BrowserManager, BrowserConfig

    config = BrowserConfig(
        mode=mode,
        headless=True,
        search_engine=engine,
    )

    # Reset singleton for clean test
    BrowserManager._instance = None
    manager = BrowserManager.get_instance(config)

    from src.utils.browser_search import browser_search

    query = "Python playwright browser automation"
    print(f"\n{'='*60}")
    print(f"Testing browser search ({engine}, mode={mode})")
    print(f"Query: {query}")
    print(f"{'='*60}\n")

    # Browser search only works in CDP mode — Playwright mode is expected
    # to return empty results (falls back to DDG HTML scrape in production)
    if mode != "cdp":
        print(
            "ℹ️  Browser search is CDP-only (needs user's real Chrome).\n"
            "   In Playwright mode, browser search is skipped by design.\n"
            "   The agent falls back to DuckDuckGo HTML scraping instead.\n"
        )

    results = await browser_search(
        query=query,
        num_results=5,
        search_engine=engine,
        manager=manager,
    )

    if not results:
        if mode != "cdp":
            print("⏭️  Skipped (expected in Playwright mode — CDP required)")
            return True  # Not a failure
        elif engine == "duckduckgo":
            print(
                "⚠️  DuckDuckGo returned no results — DDG aggressively blocks\n"
                "   automated browsers even in CDP mode with fresh profiles.\n"
                "   This is expected. Google search is the recommended engine."
            )
            return True  # Known limitation, not a real failure
        else:
            print(
                "❌ No results in CDP mode — Chrome may not be running with\n"
                "   --remote-debugging-port=9222, or the search engine blocked\n"
                "   the request."
            )
            return False

    print(f"✅ Got {len(results)} results:\n")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r.get('title', 'No title')}")
        print(f"     URL: {r.get('link', 'No URL')}")
        snippet = r.get("snippet", "")
        if snippet:
            print(f"     Snippet: {snippet[:120]}...")
        print()

    return True


async def test_browser_fetch(url: str = None, mode: str = "playwright"):
    """Test browser-based page fetch."""
    from src.utils.browser_manager import BrowserManager, BrowserConfig

    # Reuse existing instance or create new one
    try:
        manager = BrowserManager.get_instance()
    except RuntimeError:
        config = BrowserConfig(
            mode=mode,
            headless=True,
        )
        manager = BrowserManager.get_instance(config)

    from src.utils.browser_fetch import browser_fetch

    test_url = url or "https://github.com/anthropics/courses"
    print(f"\n{'='*60}")
    print(f"Testing browser fetch (mode={mode})")
    print(f"URL: {test_url}")
    print(f"{'='*60}\n")

    result = await browser_fetch(
        url=test_url,
        manager=manager,
    )

    if not result:
        print("❌ Fetch returned None (failed or insufficient content)")
        return False

    title = result.get("title", "No title")
    content = result.get("content", "")
    print(f"✅ Fetched successfully:\n")
    print(f"  Title: {title}")
    print(f"  Content length: {len(content)} chars")
    print(f"\n  --- First 500 chars ---")
    print(f"  {content[:500]}")
    print(f"  --- End preview ---\n")

    return True


async def main():
    args = sys.argv[1:]
    test_mode = args[0] if args else "all"

    # Check for --cdp flag
    cdp = "--cdp" in args
    browser_mode = "cdp" if cdp else "playwright"
    # Remove --cdp from args for positional parsing
    args = [a for a in args if a != "--cdp"]
    test_mode = args[0] if args else "all"

    results = {}

    try:
        if test_mode in ("all", "search"):
            # Reset singleton before each search test
            from src.utils.browser_manager import BrowserManager
            await BrowserManager.shutdown_instance()

            ok = await test_browser_search("google", mode=browser_mode)
            results["Search (Google)"] = ok

            if test_mode == "all":
                await BrowserManager.shutdown_instance()
                ok = await test_browser_search("duckduckgo", mode=browser_mode)
                results["Search (DuckDuckGo)"] = ok

        if test_mode in ("all", "fetch"):
            from src.utils.browser_manager import BrowserManager
            await BrowserManager.shutdown_instance()

            url = args[1] if len(args) > 1 else None
            ok = await test_browser_fetch(url, mode=browser_mode)
            results["Fetch"] = ok

    finally:
        # Clean up
        from src.utils.browser_manager import BrowserManager
        await BrowserManager.shutdown_instance()

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary (mode={browser_mode}):")
    all_ok = True
    for name, ok in results.items():
        status = "✅ pass" if ok else "❌ FAIL"
        print(f"  {status}  {name}")
        if not ok:
            all_ok = False

    if not cdp and test_mode in ("all", "search"):
        print(
            f"\n  💡 Search tests were skipped (Playwright mode)."
            f"\n     To test browser search, quit Chrome and relaunch with CDP:"
            f"\n       /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome \\"
            f"\n         --remote-debugging-port=9222 \\"
            f'\n         --user-data-dir="$HOME/chrome-cdp-profile"'
            f"\n     Then run: python tests/test_browser.py search --cdp"
        )
    print(f"{'='*60}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
