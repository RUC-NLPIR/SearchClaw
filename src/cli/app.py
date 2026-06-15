"""SearchClaw CLI entry point.

`searchclaw` launches this. On first run (or when credentials are missing) it
runs a plain-stdin setup wizard, then hands off to the full-screen Textual TUI
(src/cli/tui/app.py) which drives the same query_loop agent core the web
server uses.
"""

from __future__ import annotations

import asyncio
import logging
import os

# Disable Textual's Kitty keyboard protocol BEFORE any textual import.
# With it on, CJK IME input arrives as raw `CSI ...u` sequences (e.g. typing
# 你好 yields "[32;;20320:22909u") because Textual 8.x doesn't decode the
# protocol's associated-text payload. Turning it off routes IME input through
# the normal text path. textual.constants reads this at import time, so it
# must be set first.
os.environ.setdefault("TEXTUAL_DISABLE_KITTY_KEY", "1")

# Use litellm's bundled model-cost table instead of fetching it from GitHub on
# import. Without this, environments that can't reach raw.githubusercontent.com
# print an SSL/timeout WARNING before falling back to the local copy anyway.
# litellm reads this at import time, so it must precede any litellm import.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from rich.console import Console

from src.cli import config as cli_config
from src.cli.runtime import build_runtime
from src.cli.theme import THEME

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)
# pdfminer (under pdfplumber) logs benign per-font warnings such as "Could not
# get FontBBox …" when a PDF's font descriptor is malformed. Text extraction
# still works; silence them so they don't corrupt the full-screen TUI.
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)

console = Console(theme=THEME)


async def _async_main() -> None:
    cfg = cli_config.load_cli_config()
    if cfg is None or not cli_config.has_llm_credentials(cfg):
        if cfg is None:
            console.print("[bold]Welcome to SearchClaw![/bold] Let's set things up.")
        else:
            console.print("[yellow]No LLM credentials found.[/yellow] Let's complete setup.")
        cfg = cli_config.run_setup_wizard(cfg)

    cli_config.apply_env_from_config(cfg)
    runtime = build_runtime(cfg)
    # The setup wizard above runs on plain stdin/stdout; the interactive REPL
    # is a full-screen Textual TUI.
    from src.cli.tui.app import SearchClawApp
    app = SearchClawApp(runtime)
    try:
        await app.run_async()
    finally:
        await runtime.aclose()


def main() -> None:
    """Console-script entry point (`searchclaw`)."""
    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
