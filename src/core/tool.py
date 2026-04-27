"""
Tool base class and registry.

Every tool implements a common interface with input validation,
execution, and metadata (concurrency safety, result size limits, etc.).
The ToolRegistry collects all tools and provides lookup.

Key design decisions:
- Fail-closed defaults: is_concurrency_safe=False, is_read_only=True
- Each tool contributes to the system prompt via prompt()
- max_result_size_chars prevents context blowup from large pages
- input_schema uses JSON Schema (compatible with LLM function calling)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.types import ToolResult, ValidationResult

logger = logging.getLogger(__name__)

# Directory for caching oversized tool results.
# Configurable via settings.yaml tools.cache_dir (default: ./cache).
# Resolved at startup by router.py and passed through ToolUseContext.
CACHE_DIR = Path("./cache")


@dataclass
class ToolUseContext:
    """
    Context passed to every tool call.

    Carries session state and references to shared resources.
    """
    session_id: str = ""
    turn_count: int = 0
    cache_dir: Path = field(default_factory=lambda: CACHE_DIR)
    # Rate limiter reference (injected by the loop)
    rate_limiter: Any = None
    # Extra context tools might need
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)


class Tool(ABC):
    """
    Base class for all tools in the search agent.

    Each tool:
    - Has a name and JSON Schema for its inputs
    - Contributes a prompt() section to the system prompt
    - Declares concurrency safety and result size limits
    - Executes via call() and returns a ToolResult
    """

    # --- Required attributes (set by subclasses) ---
    name: str = ""
    description: str = ""

    # JSON Schema for the tool's input parameters.
    # Used by the LLM for function calling.
    # Subclasses MUST override this with their own dict.
    input_schema: dict

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Ensure each subclass gets its own input_schema dict
        # to prevent accidental sharing of the mutable default.
        if "input_schema" not in cls.__dict__:
            cls.input_schema = {}

    # --- Safety & concurrency metadata ---

    # Can this tool run in parallel with other instances?
    # Default False (fail-closed). Search tools should override to True.
    is_concurrency_safe: bool = False

    # Does this tool only read data (no side effects)?
    # All search tools are read-only.
    is_read_only: bool = True

    # Maximum characters in tool result before caching to disk.
    # Prevents context blowup from large web pages.
    max_result_size_chars: int = 50000

    # --- Methods ---

    @abstractmethod
    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        """
        Execute this tool with the given arguments.

        Returns a ToolResult containing the output data, any citations
        discovered, and whether the result was truncated.
        """
        ...

    def prompt(self) -> str:
        """
        System prompt contribution.

        Tells the LLM when and how to use this tool. Appended to the
        system prompt by the ContextBuilder. Override for custom guidance.
        """
        return ""

    def validate_input(self, args: dict) -> ValidationResult:
        """
        Validate tool input before execution.

        Override for tool-specific validation (e.g., URL format, query length).
        Default: always valid.
        """
        return ValidationResult(valid=True)

    async def aclose(self) -> None:
        """
        Close any resources held by this tool (e.g., httpx clients).

        Called during application shutdown. Subclasses with HTTP clients
        don't need to override this — the default implementation closes
        any httpx.AsyncClient found on standard attribute names.
        """
        for attr in ("_client", "_jina_client"):
            client = getattr(self, attr, None)
            if client is not None and hasattr(client, "aclose"):
                try:
                    await client.aclose()
                except Exception:
                    pass

    def to_api_schema(self) -> dict:
        """
        Convert to the format expected by litellm/OpenAI function calling API.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    async def _maybe_truncate(self, data: str, url: str, context: ToolUseContext) -> tuple[str, bool, str | None]:
        """
        If data exceeds max_result_size_chars, cache to disk and return preview.

        Oversized results are persisted to disk with a preview + path
        in the context, so the agent can use deep_read to access them.
        """
        if len(data) <= self.max_result_size_chars:
            return data, False, None

        # Cache full content to disk
        import hashlib
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        cached_path = context.cache_dir / f"{self.name}_{url_hash}.md"
        cached_path.write_text(data, encoding="utf-8")

        # Return truncated preview
        preview = data[: self.max_result_size_chars]
        preview += f"\n\n---\n[Content truncated. Full content ({len(data):,} chars) saved to: {cached_path}]"

        logger.info(f"Tool {self.name}: cached {len(data):,} chars to {cached_path}")
        return preview, True, str(cached_path)


class ToolRegistry:
    """
    Collects all available tools and provides lookup.

    Single source of truth for which tools are available. Tools are
    registered at startup and looked up by name during the agentic loop.
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool. Last registration wins on name collision."""
        if tool.name in self._tools:
            logger.warning(f"Tool '{tool.name}' already registered, overwriting")
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def get(self, name: str) -> Tool | None:
        """Look up a tool by name."""
        return self._tools.get(name)

    def remove(self, name: str) -> None:
        """Remove a tool by name (no-op if not registered)."""
        if name in self._tools:
            del self._tools[name]
            logger.info(f"Removed tool: {name}")

    def all_tools(self) -> list[Tool]:
        """All registered tools (for system prompt building)."""
        return list(self._tools.values())

    def get_api_schemas(self) -> list[dict]:
        """All tool schemas for LLM function calling API."""
        return [tool.to_api_schema() for tool in self._tools.values()]

    def get_concurrent_safe(self) -> set[str]:
        """Names of tools that can run in parallel."""
        return {name for name, tool in self._tools.items() if tool.is_concurrency_safe}

    async def close_all(self) -> None:
        """Close any HTTP clients held by registered tools."""
        for tool in self._tools.values():
            try:
                await tool.aclose()
            except Exception:
                pass


def build_default_registry(config: dict | None = None) -> ToolRegistry:
    """
    Build the default tool registry with all search tools.

    Called once at startup.

    Args:
        config: Optional dict of tool configuration from settings.yaml.
                Keys like "web_search_default_results", "http_timeout", etc.
    """
    from src.tools.web_search import WebSearchTool
    from src.tools.web_fetch import WebFetchTool
    from src.tools.deep_read import DeepReadTool
    from src.tools.cite_source import CiteSourceTool
    from src.tools.research_plan import ResearchPlanTool
    from src.tools.ask_user import AskUserTool

    cfg = config or {}

    registry = ToolRegistry()
    registry.register(WebSearchTool(
        default_results=cfg.get("web_search_default_results", 10),
        max_results=cfg.get("web_search_max_results", 20),
        max_result_size_chars=cfg.get("max_result_size_chars", 15000),
        http_timeout=cfg.get("http_timeout", 30),
    ))
    registry.register(WebFetchTool(
        max_result_size_chars=cfg.get("max_result_size_chars", 50000),
        http_timeout=cfg.get("http_timeout", 30),
        jina_timeout=cfg.get("jina_timeout", 60),
        extraction_threshold=cfg.get("content_extraction_threshold", 15000),
    ))
    registry.register(DeepReadTool(
        max_result_size_chars=cfg.get("max_result_size_chars", 30000),
    ))
    registry.register(CiteSourceTool())
    registry.register(ResearchPlanTool())
    registry.register(AskUserTool())

    # Optional tools — register only if dependencies are available
    try:
        from src.tools.academic_search import AcademicSearchTool
        registry.register(AcademicSearchTool(
            default_results=cfg.get("academic_search_default_results", 5),
            max_results=cfg.get("academic_search_max_results", 10),
            max_result_size_chars=cfg.get("max_result_size_chars", 20000),
            http_timeout=cfg.get("http_timeout", 30),
        ))
    except ImportError:
        logger.info("Academic search tool not available (missing dependencies)")

    try:
        from src.tools.news_search import NewsSearchTool
        registry.register(NewsSearchTool(
            default_results=cfg.get("news_search_default_results", 5),
            max_results=cfg.get("news_search_max_results", 10),
            default_days_back=cfg.get("news_search_default_days_back", 7),
            max_days_back=cfg.get("news_search_max_days_back", 30),
            max_result_size_chars=cfg.get("max_result_size_chars", 15000),
            http_timeout=cfg.get("http_timeout", 30),
        ))
    except ImportError:
        logger.info("News search tool not available (missing dependencies)")

    try:
        from src.tools.wechat_search import WeChatSearchTool
        registry.register(WeChatSearchTool(
            http_timeout=cfg.get("http_timeout", 15),
            max_result_size_chars=cfg.get("max_result_size_chars", 30000),
        ))
    except Exception:
        logger.warning("WeChat search tool failed to initialize", exc_info=True)

    try:
        from src.tools.exa_search import ExaSearchTool
        registry.register(ExaSearchTool(
            default_results=cfg.get("exa_search_default_results", 10),
            max_results=cfg.get("exa_search_max_results", 20),
            max_result_size_chars=cfg.get("max_result_size_chars", 15000),
            http_timeout=cfg.get("http_timeout", 30),
        ))
    except ImportError:
        logger.info("Exa search tool not available (missing dependencies)")

    # --- Browser integration (optional) ---
    # Wire up BrowserManager to web_search and web_fetch tools when enabled.
    # The browser is NOT a separate tool — it's a backend fallback within
    # existing tools.  The LLM doesn't know or care whether results came
    # from Serper/Jina or from a browser.
    browser_search_enabled = cfg.get("browser_search_enabled", False)
    browser_fetch_enabled = cfg.get("browser_fetch_enabled", False)

    if browser_search_enabled or browser_fetch_enabled:
        try:
            from src.utils.browser_manager import BrowserManager, BrowserConfig

            browser_config = BrowserConfig(
                mode=cfg.get("browser_mode", "playwright"),
                cdp_port=int(cfg.get("browser_cdp_port", 9222)),
                chrome_path=cfg.get("browser_chrome_path", ""),
                headless=cfg.get("browser_headless", True),
                user_data_dir=cfg.get("browser_user_data_dir", ""),
                search_engine=cfg.get("browser_search_engine", "google"),
                use_for_search=browser_search_enabled,
                use_for_fetch=browser_fetch_enabled,
            )
            browser_manager = BrowserManager.get_instance(browser_config)

            # Inject browser manager into tools that support it
            if browser_search_enabled:
                web_search_tool = registry.get("web_search")
                if web_search_tool:
                    web_search_tool._browser_manager = browser_manager
                    web_search_tool._search_engine = browser_config.search_engine
                    logger.info("Browser search enabled as web_search fallback")

            if browser_fetch_enabled:
                web_fetch_tool = registry.get("web_fetch")
                if web_fetch_tool:
                    web_fetch_tool._browser_manager = browser_manager
                    logger.info("Browser fetch enabled as web_fetch fallback")

        except ImportError:
            logger.warning(
                "Browser integration enabled in config but playwright is not "
                "installed. Install with: pip install 'search-agent[browser]' "
                "&& playwright install chromium"
            )

    return registry
