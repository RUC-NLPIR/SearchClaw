"""CLI configuration — user-level config at ~/.searchclaw/config.yaml.

The CLI keeps its own config separate from the project's
config/settings.yaml so that `searchclaw` works from any directory after
a one-time setup. The wizard collects the LLM endpoint and the API keys
that tools read from the environment, then writes a settings dict that
mirrors the structure of config/settings.yaml (so runtime.build_runtime
can consume it identically to the web path).

API keys are stored under a top-level `secrets:` block — an arbitrary map
of ENV_VAR -> value. apply_env_from_config() exports every entry to
os.environ before the tool registry is built. litellm reads the provider
key for the active model (ANTHROPIC_API_KEY, OPENAI_API_KEY,
DEEPSEEK_API_KEY, …) and tools read SERPER_API_KEY / JINA_API_KEY /
NEWSAPI_KEY straight from the environment. Because the block is generic,
any custom proxy / provider key works without code changes.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path.home() / ".searchclaw"
CONFIG_PATH = CONFIG_DIR / "config.yaml"
HISTORY_PATH = CONFIG_DIR / "history"

# Search/fetch tool keys the wizard always offers (independent of the LLM).
TOOL_ENV_KEYS = (
    ("SERPER_API_KEY", "web search; blank = DuckDuckGo fallback"),
    ("JINA_API_KEY", "better page fetch; blank = direct fetch"),
    ("NEWSAPI_KEY", "news search; blank = Google News RSS"),
)

# A valid environment variable name (for the custom-vars loop).
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def provider_env_var(model: str) -> str:
    """Map a litellm model id to the env var litellm reads for its API key.

    e.g. 'deepseek/deepseek-v4-pro' -> 'DEEPSEEK_API_KEY',
         'anthropic/claude-opus-4.6' -> 'ANTHROPIC_API_KEY',
         'openai/gpt-5' -> 'OPENAI_API_KEY'.
    Falls back to OPENAI_API_KEY (the common case for OpenAI-compatible
    proxies addressed via the 'openai/' prefix).
    """
    prefix = model.split("/", 1)[0].strip().lower() if "/" in model else ""
    known = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "groq": "GROQ_API_KEY",
        "xai": "XAI_API_KEY",
        "together_ai": "TOGETHERAI_API_KEY",
    }
    if prefix in known:
        return known[prefix]
    if prefix:
        return f"{prefix.upper()}_API_KEY"
    return "OPENAI_API_KEY"



def config_exists() -> bool:
    return CONFIG_PATH.exists()


def load_cli_config() -> dict[str, Any] | None:
    """Load the CLI config, or None if it doesn't exist / is unreadable."""
    if not CONFIG_PATH.exists():
        return None
    try:
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return None


def apply_env_from_config(cfg: dict[str, Any], force: bool = False) -> None:
    """Export every entry in the config's `secrets:` block to os.environ.

    Must run BEFORE build_runtime() — tools and litellm read these from the
    environment at construction / call time. The block is a generic
    ENV_VAR -> value map, so any provider/proxy key the user adds is
    exported without code changes.

    By default an existing environment value takes precedence (a key the
    user set in their shell wins over the config). Pass force=True on a
    /config reload: the user just edited the keys in the wizard, so a
    non-empty config value should overwrite whatever a previous reload (or
    the config itself) had exported.
    """
    secrets = cfg.get("secrets", {}) or {}
    for key, raw in secrets.items():
        val = str(raw or "").strip()
        if not val:
            continue
        if force or not os.environ.get(key):
            os.environ[key] = val



def _default_settings() -> dict[str, Any]:
    """Settings skeleton mirroring config/settings.yaml defaults.

    The wizard fills in llm.base_url / llm.default_model and the secrets
    block; everything else uses sane defaults that match the project file.
    """
    return {
        "llm": {
            "default_model": "anthropic/claude-opus-4.6",
            "side_query_model": "anthropic/claude-sonnet-4.6",
            "fallback_model": "anthropic/claude-sonnet-4.6",
            "max_tokens": 128000,
            "base_url": "",
            "side_query_base_url": "",
            "reasoning_effort": "max",
            "stream": True,
            "max_retries": 3,
            "retry_base_delay_ms": 500,
        },
        "limits": {
            "max_turns": 100,
            "max_search": 50,
            "max_fetch": 50,
            "compact_threshold_tokens": 80000,
            "rate_limit_per_domain": 50,
        },
        "tools": {
            "cache_dir": str(CONFIG_DIR / "cache"),
            "web_search_default_results": 10,
            "web_search_max_results": 20,
            "academic_search_default_results": 5,
            "academic_search_max_results": 10,
            "news_search_default_results": 5,
            "news_search_max_results": 10,
            "news_search_default_days_back": 7,
            "news_search_max_days_back": 30,
            "max_result_size_chars": 50000,
            "http_timeout": 30,
            "jina_timeout": 60,
            "content_extraction_threshold": 15000,
        },
        "skills": {
            "enabled": True,
            "dirs": ["./skills"],
            "listing_max_chars": 8000,
            "max_skill_chars": 50000,
            "script_timeout_seconds": 30,
            "script_max_output_chars": 20000,
        },
        "hooks": {
            "min_citations": 0,
            "min_domains": 2,
            "min_answer_chars": 200,
        },
        "memory": {
            "enabled": True,
            "base_dir": str(CONFIG_DIR / "memory"),
            "max_relevant_memories": 5,
        },
        "secrets": {},
    }


def _prompt(label: str, default: str = "", secret: bool = False) -> str:
    """Prompt for a single value with an optional default."""
    import getpass

    suffix = f" [{default}]" if default else ""
    if secret:
        raw = getpass.getpass(f"{label}{suffix}: ").strip()
    else:
        raw = input(f"{label}{suffix}: ").strip()
    return raw or default


def run_setup_wizard(existing: dict[str, Any] | None = None) -> dict[str, Any]:
    """Interactively collect config and write it to ~/.searchclaw/config.yaml.

    Returns the saved settings dict. Re-runnable via the /config command;
    when `existing` is provided its values are offered as defaults.
    """
    base = existing or _default_settings()
    # Ensure all default sections exist even when editing an old config.
    defaults = _default_settings()
    for section, vals in defaults.items():
        base.setdefault(section, vals if not isinstance(vals, dict) else dict(vals))

    llm = base["llm"]
    secrets = base.setdefault("secrets", {})

    print("\n  SearchClaw setup")
    print("  ----------------")
    print("  Configure your LLM provider and search API keys.")
    print("  Press Enter to accept the [default]. Leave keys blank to skip.\n")

    print("  LLM provider")
    llm["base_url"] = _prompt(
        "  Base URL (custom/proxy endpoint; blank = provider default)",
        llm.get("base_url", ""),
    )
    llm["default_model"] = _prompt(
        "  Default model (litellm id, e.g. anthropic/claude-opus-4.6)",
        llm.get("default_model", "anthropic/claude-opus-4.6"),
    )
    llm["side_query_model"] = _prompt(
        "  Side-query model (cheap model for ranking/quality)",
        llm.get("side_query_model", "anthropic/claude-sonnet-4.6"),
    )
    llm["fallback_model"] = _prompt(
        "  Fallback model (used if the default model errors)",
        llm.get("fallback_model", llm["default_model"]),
    )

    print("\n  Provider API keys (stored in ~/.searchclaw/config.yaml)")
    # Derive the exact env var litellm needs from each configured model, so
    # the user is asked for precisely the right key (e.g. DEEPSEEK_API_KEY
    # for deepseek/* models) instead of a fixed Anthropic/OpenAI pair.
    provider_vars: list[str] = []
    for model in (llm["default_model"], llm["side_query_model"], llm["fallback_model"]):
        var = provider_env_var(model)
        if var not in provider_vars:
            provider_vars.append(var)
    if llm["base_url"]:
        print("  (Using a custom base_url? Enter the key your proxy expects,")
        print("   or leave blank if the proxy needs no key.)")
    for var in provider_vars:
        secrets[var] = _prompt(f"  {var}", secrets.get(var, ""), secret=True)

    print("\n  Search / fetch API keys (optional)")
    for var, hint in TOOL_ENV_KEYS:
        secrets[var] = _prompt(f"  {var} ({hint})", secrets.get(var, ""), secret=True)

    print("\n  Extra environment variables (optional)")
    print("  Add any other key your provider/proxy needs (e.g. a custom auth")
    print("  var). Enter a NAME, then its value. Blank name to finish.")
    while True:
        name = _prompt("  Var name (blank to finish)", "")
        if not name:
            break
        if not _ENV_NAME_RE.match(name):
            print(f"    '{name}' is not a valid env var name — skipped.")
            continue
        secrets[name] = _prompt(f"  {name}", secrets.get(name, ""), secret=True)

    # Drop empty entries so the saved file stays clean.
    base["secrets"] = {k: v for k, v in secrets.items() if str(v or "").strip()}

    save_config(base)
    print(f"\n  Saved config to {CONFIG_PATH}\n")
    return base


def save_config(cfg: dict[str, Any]) -> Path:
    """Write the config to disk with 0600 permissions (it holds secrets)."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    try:
        os.chmod(CONFIG_PATH, 0o600)
    except OSError:
        pass
    return CONFIG_PATH


def has_llm_credentials(cfg: dict[str, Any]) -> bool:
    """True if a usable LLM credential is available.

    A custom base_url (local proxy / vLLM) often needs no provider key, so
    that counts. Otherwise we need the provider key for the default model —
    derived from its prefix — present in the config's secrets or the
    ambient environment.
    """
    llm = cfg.get("llm", {}) or {}
    if str(llm.get("base_url", "") or "").strip():
        return True
    secrets = cfg.get("secrets", {}) or {}
    var = provider_env_var(llm.get("default_model", ""))
    if str(secrets.get(var, "") or "").strip() or os.environ.get(var):
        return True
    # Also accept the two most common keys, in case the model id is unusual.
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        if str(secrets.get(key, "") or "").strip() or os.environ.get(key):
            return True
    return False
