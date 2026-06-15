"""Shared runtime assembly for SearchClaw consumers.

Builds the full set of shared resources (LLM client, tool registry, hook
engine, memory store, session storage, rate limiter, context builder) from
a settings dict. This mirrors the module-level wiring in src/web/router.py
so the CLI can construct an identical agent runtime in-process.

The settings dict has the same shape as config/settings.yaml (llm, limits,
tools, hooks, memory, skills blocks). API keys are expected to already be
in os.environ (see cli.config.apply_env_from_config).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.context import ContextBuilder
from src.core.tool import ToolRegistry, build_default_registry
from src.hooks.builtin_hooks import build_default_hooks
from src.hooks.engine import HookEngine
from src.hooks.plan_completeness_hook import PlanCompletenessHook
from src.llm.client import LLMClient, ModelConfig, set_shared_config
from src.memory.store import MemoryStore
from src.utils.rate_limiter import DomainRateLimiter

logger = logging.getLogger(__name__)


@dataclass
class Runtime:
    """Bundle of shared resources needed to run the agent loop."""

    model_config: ModelConfig
    llm_client: LLMClient
    tool_registry: ToolRegistry
    hook_engine: HookEngine
    memory_store: MemoryStore
    rate_limiter: DomainRateLimiter
    context_builder: ContextBuilder
    # Resolved limits / paths pulled from settings, for QueryParams.
    max_turns: int = 100
    max_search: int = 50
    max_fetch: int = 50
    compact_threshold_tokens: int = 80000
    cache_dir: str = "./cache"
    memory_enabled: bool = True
    max_relevant_memories: int = 5

    async def aclose(self) -> None:
        """Release HTTP clients held by tools (mirrors router lifespan)."""
        try:
            await self.tool_registry.close_all()
        except Exception:
            pass


def _model_config_from_settings(settings: dict[str, Any]) -> ModelConfig:
    """Build a ModelConfig from the `llm:` block of a settings dict.

    ModelConfig.from_settings reads a YAML file; here we already have the
    dict in hand (from ~/.searchclaw/config.yaml), so map it directly.
    """
    llm = settings.get("llm", {}) or {}
    defaults = ModelConfig()
    return ModelConfig(
        default_model=llm.get("default_model", defaults.default_model),
        side_query_model=llm.get("side_query_model", defaults.side_query_model),
        fallback_model=llm.get("fallback_model", defaults.fallback_model),
        max_tokens=int(llm.get("max_tokens", defaults.max_tokens)),
        base_url=llm.get("base_url", "") or "",
        side_query_base_url=llm.get("side_query_base_url", "") or "",
        max_retries=int(llm.get("max_retries", defaults.max_retries)),
        retry_base_delay_ms=int(llm.get("retry_base_delay_ms", defaults.retry_base_delay_ms)),
        reasoning_effort=llm.get("reasoning_effort", "") or "",
        stream=bool(llm.get("stream", defaults.stream)),
    )


def build_runtime(settings: dict[str, Any]) -> Runtime:
    """Assemble all shared resources from a settings dict.

    Reuses the same building blocks as src/web/router.py:
    build_default_registry, build_default_hooks, PlanCompletenessHook, and
    the skills loader.
    """
    model_config = _model_config_from_settings(settings)
    set_shared_config(model_config)  # side_query() picks this up globally

    limits = settings.get("limits", {}) or {}
    tools_cfg = settings.get("tools", {}) or {}
    hooks_cfg = settings.get("hooks", {}) or {}
    memory_cfg = settings.get("memory", {}) or {}
    skills_cfg = settings.get("skills", {}) or {}
    browser_cfg = settings.get("browser", {}) or {}

    max_turns = int(limits.get("max_turns", 100))
    max_search = int(limits.get("max_search", 50))
    max_fetch = int(limits.get("max_fetch", 50))
    compact_threshold_tokens = int(limits.get("compact_threshold_tokens", 80000))
    rate_limit_per_domain = int(limits.get("rate_limit_per_domain", 50))
    cache_dir = tools_cfg.get("cache_dir", "./cache")

    browser_enabled = bool(browser_cfg.get("enabled", False))
    browser_use_for_search = bool(browser_cfg.get("use_for_search", True))
    browser_use_for_fetch = bool(browser_cfg.get("use_for_fetch", True))

    tool_registry = build_default_registry(config={
        "web_search_default_results": int(tools_cfg.get("web_search_default_results", 10)),
        "web_search_max_results": int(tools_cfg.get("web_search_max_results", 20)),
        "academic_search_default_results": int(tools_cfg.get("academic_search_default_results", 5)),
        "academic_search_max_results": int(tools_cfg.get("academic_search_max_results", 10)),
        "news_search_default_results": int(tools_cfg.get("news_search_default_results", 5)),
        "news_search_max_results": int(tools_cfg.get("news_search_max_results", 10)),
        "news_search_default_days_back": int(tools_cfg.get("news_search_default_days_back", 7)),
        "news_search_max_days_back": int(tools_cfg.get("news_search_max_days_back", 30)),
        "max_result_size_chars": int(tools_cfg.get("max_result_size_chars", 50000)),
        "http_timeout": int(tools_cfg.get("http_timeout", 30)),
        "jina_timeout": int(tools_cfg.get("jina_timeout", 60)),
        "content_extraction_threshold": int(tools_cfg.get("content_extraction_threshold", 15000)),
        "browser_search_enabled": browser_enabled and browser_use_for_search,
        "browser_fetch_enabled": browser_enabled and browser_use_for_fetch,
        "browser_mode": browser_cfg.get("mode", "playwright"),
        "browser_cdp_port": int(browser_cfg.get("cdp_port", 9222)),
        "browser_chrome_path": browser_cfg.get("chrome_path", ""),
        "browser_headless": bool(browser_cfg.get("headless", True)),
        "browser_user_data_dir": browser_cfg.get("user_data_dir", ""),
        "browser_search_engine": browser_cfg.get("search_engine", "google"),
    })

    # Skills (same pattern as router.py:193-219)
    if bool(skills_cfg.get("enabled", True)):
        try:
            from src.skills.loader import load_skills
            from src.tools.run_skill_script import RunSkillScriptTool
            from src.tools.use_skill import UseSkillTool

            configured_dirs = skills_cfg.get("dirs", ["./skills"])
            if isinstance(configured_dirs, str):
                skill_dirs = [d.strip() for d in configured_dirs.split(",") if d.strip()]
            else:
                skill_dirs = [str(d) for d in configured_dirs]

            loaded_skills = load_skills(skill_dirs, root=Path.cwd())
            tool_registry.register(UseSkillTool(
                skills=loaded_skills,
                listing_max_chars=int(skills_cfg.get("listing_max_chars", 8000)),
                max_skill_chars=int(skills_cfg.get("max_skill_chars", 50000)),
            ))
            tool_registry.register(RunSkillScriptTool(
                skills=loaded_skills,
                default_timeout_seconds=int(skills_cfg.get("script_timeout_seconds", 30)),
                max_output_chars=int(skills_cfg.get("script_max_output_chars", 20000)),
            ))
            logger.info("Skills: loaded %d", len(loaded_skills))
        except Exception:
            logger.warning("Skills: failed to initialize", exc_info=True)

    # Hook engine (same pattern as router.py:258-270)
    hook_engine = HookEngine()
    for hook in build_default_hooks(config={
        "min_citations": int(hooks_cfg.get("min_citations", 2)),
        "min_domains": int(hooks_cfg.get("min_domains", 2)),
        "min_answer_chars": int(hooks_cfg.get("min_answer_chars", 200)),
    }):
        hook_engine.register_stop_hook(hook)
    hook_engine.register_stop_hook(PlanCompletenessHook())

    memory_store = MemoryStore(base_dir=memory_cfg.get("base_dir", "./memory"))

    return Runtime(
        model_config=model_config,
        llm_client=LLMClient(config=model_config),
        tool_registry=tool_registry,
        hook_engine=hook_engine,
        memory_store=memory_store,
        rate_limiter=DomainRateLimiter(max_per_minute=rate_limit_per_domain),
        context_builder=ContextBuilder(),
        max_turns=max_turns,
        max_search=max_search,
        max_fetch=max_fetch,
        compact_threshold_tokens=compact_threshold_tokens,
        cache_dir=cache_dir,
        memory_enabled=bool(memory_cfg.get("enabled", True)),
        max_relevant_memories=int(memory_cfg.get("max_relevant_memories", 5)),
    )
