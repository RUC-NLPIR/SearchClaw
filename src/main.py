"""
SearchClaw — FastAPI entry point.

Starts the web server with WebSocket support.

Usage:
    python -m src.main
    # or
    uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import sys

import uvicorn

from src.web.router import create_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Create the FastAPI app
app = create_app()


def main():
    """Run the search agent server."""
    # Read host/port from settings.yaml, env vars override
    _host = "127.0.0.1"
    _port = 8000
    _api_key = ""
    _cfg = {}
    try:
        import yaml
        with open("config/settings.yaml") as f:
            _cfg = yaml.safe_load(f) or {}
        _srv = _cfg.get("server", {})
        _host = _srv.get("host", "127.0.0.1")
        _port = int(_srv.get("port", 8000))
        _api_key = os.environ.get("SEARCH_CLAW_API_KEY", "") or _srv.get("api_key", "")
    except Exception:
        pass

    host = os.environ.get("HOST", _host)
    port = int(os.environ.get("PORT", str(_port)))

    logger.info(f"Starting SearchClaw on {host}:{port}")
    logger.info(f"Open http://localhost:{port} in your browser")

    # Warn about unauthenticated non-localhost deployments
    if host != "127.0.0.1" and host != "localhost" and not _api_key:
        logger.warning(
            f"Server binding to {host} WITHOUT authentication. "
            "This is dangerous — anyone on the network can access the agent. "
            "Set 'server.api_key' in settings.yaml or SEARCH_CLAW_API_KEY env var."
        )

    # Check for API keys
    if not os.environ.get("SERPER_API_KEY"):
        logger.warning(
            "SERPER_API_KEY not set — web search will use DuckDuckGo fallback. "
            "Set SERPER_API_KEY for better results."
        )

    if not os.environ.get("JINA_API_KEY"):
        logger.info(
            "JINA_API_KEY not set — web_fetch will use Jina Reader in free mode "
            "(20 RPM rate limit). Set JINA_API_KEY for 200 RPM."
        )

    if os.environ.get("EXA_API_KEY"):
        logger.info("Exa search enabled with API key (higher rate limits)")
    else:
        logger.info(
            "EXA_API_KEY not set — exa_search will use Exa MCP endpoint directly "
            "(rate-limited). Set EXA_API_KEY for higher limits."
        )

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        logger.error(
            "No LLM API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY. "
            "The agent will not work without an LLM provider."
        )
        sys.exit(1)

    # Browser integration status
    _browser_enabled_cfg = _cfg.get("browser", {}).get("enabled", False)

    if _browser_enabled_cfg:
        try:
            import playwright
            logger.info(
                "Browser integration enabled (Playwright). "
                "Browser will launch on first search/fetch request."
            )
        except ImportError:
            logger.warning(
                "browser.enabled=true in settings.yaml but playwright is not installed. "
                "Install with: pip install 'search-agent[browser]' && playwright install chromium"
            )

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
