"""
LLM client wrapper around litellm.

Provides streaming calls for the main loop and quick side-queries
for routing/ranking tasks.

All model names and base URLs are configurable via config/settings.yaml
or by passing a ModelConfig directly.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator

import litellm

from src.core.types import EventType, StreamEvent

logger = logging.getLogger(__name__)

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True
# Automatically drop unsupported params per model (e.g., max_tokens vs
# max_completion_tokens) instead of raising errors.
litellm.drop_params = True


def _is_retryable(error: Exception) -> bool:
    """
    Decide whether an LLM error is transient and worth retrying.

    Retries on rate limits, overload, connection errors, server errors.
    Does NOT retry on 400 Bad Request (malformed request won't fix
    itself on retry).
    """
    error_str = str(error).lower()
    error_type = type(error).__name__

    # Connection errors — proxy down, network blip
    if "connection" in error_type.lower() or "connect" in error_str:
        return True

    # Rate limits (429)
    if "ratelimit" in error_type.lower() or "429" in error_str:
        return True

    # Overloaded (529) / server errors (5xx)
    if "overloaded" in error_str or "529" in error_str:
        return True
    if "internalservererror" in error_type.lower() or "500" in error_str:
        # But NOT if it's wrapping a connection error to localhost
        # (that's a permanent "proxy is down" situation, not a transient 500)
        if "connect call failed" in error_str or "cannot connect" in error_str:
            return True
        return True

    # Timeout
    if "timeout" in error_str or "408" in error_str:
        return True

    # NOT retryable: BadRequest (400), AuthenticationError (401/403),
    # NotFoundError (404), UnsupportedParamsError, etc.
    return False


@dataclass
class ModelConfig:
    """
    Configuration for LLM models.

    All fields can be set via config/settings.yaml under the `llm:` key,
    or via environment variables, or by passing values directly.

    base_url: custom API endpoint (vLLM, Ollama, LiteLLM proxy, Azure, etc.)
              When set, litellm sends all requests here instead of the
              provider's default URL. Leave empty to use provider defaults.
    side_query_base_url: separate endpoint for side-query model.
              Falls back to base_url if empty.
    """
    default_model: str = "anthropic/claude-sonnet-4-20250514"
    side_query_model: str = "anthropic/claude-haiku-3-20250305"
    fallback_model: str = "openai/gpt-4o-mini"
    max_tokens: int = 4096
    base_url: str = ""
    side_query_base_url: str = ""
    max_retries: int = 3
    retry_base_delay_ms: int = 500

    @classmethod
    def from_settings(cls, settings_path: str | Path = "config/settings.yaml") -> ModelConfig:
        """
        Load ModelConfig from settings.yaml.

        Falls back to defaults if the file doesn't exist or is malformed.
        """
        path = Path(settings_path)
        if not path.exists():
            logger.info(f"Settings file {path} not found, using defaults")
            return cls()

        try:
            import yaml
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            llm = data.get("llm", {})
            return cls(
                default_model=llm.get("default_model", cls.default_model),
                side_query_model=llm.get("side_query_model", cls.side_query_model),
                fallback_model=llm.get("fallback_model", cls.fallback_model),
                max_tokens=int(llm.get("max_tokens", cls.max_tokens)),
                base_url=llm.get("base_url", "") or "",
                side_query_base_url=llm.get("side_query_base_url", "") or "",
                max_retries=int(llm.get("max_retries", cls.max_retries)),
                retry_base_delay_ms=int(llm.get("retry_base_delay_ms", cls.retry_base_delay_ms)),
            )
        except Exception as e:
            logger.warning(f"Failed to load settings from {path}: {e}, using defaults")
            return cls()


class LLMClient:
    """
    Wrapper around litellm for streaming LLM calls.

    Handles:
    - Streaming responses with tool calling
    - Model fallback on failure
    """

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()

    async def stream(
        self,
        messages: list[dict],
        system_prompt: str,
        tools: list[dict] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream an LLM response, yielding StreamEvents.

        Retries transient errors (rate limits, overload, connection) with
        exponential backoff, then falls back to fallback model. Permanent
        errors (400 Bad Request) fail immediately.
        """
        target_model = model or self.config.default_model
        target_max_tokens = max_tokens or self.config.max_tokens

        # Build the messages list with system prompt
        api_messages = [{"role": "system", "content": system_prompt}] + messages

        last_error: Exception | None = None

        for attempt in range(1, self.config.max_retries + 2):  # +1 for the initial attempt
            try:
                # Build litellm kwargs
                completion_kwargs: dict = {
                    "model": target_model,
                    "messages": api_messages,
                    "max_tokens": target_max_tokens,
                    "max_completion_tokens": target_max_tokens,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }

                # Only include tools if provided — Anthropic rejects
                # tools=None when conversation history contains tool calls
                if tools:
                    completion_kwargs["tools"] = tools

                # Apply base_url if configured
                base_url = self.config.base_url
                if base_url:
                    completion_kwargs["api_base"] = base_url

                response = await litellm.acompletion(**completion_kwargs)

                # Accumulators for the streamed response
                current_tool_call: dict | None = None
                tool_call_args_buffer = ""

                async for chunk in response:
                    delta = chunk.choices[0].delta if chunk.choices else None

                    if delta is None:
                        continue

                    # Text content
                    if delta.content:
                        yield StreamEvent(
                            type=EventType.TEXT_DELTA,
                            data={"text": delta.content},
                        )

                    # Tool calls
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            if tc.function and tc.function.name:
                                # New tool call starting
                                if current_tool_call and tool_call_args_buffer:
                                    # Emit the previous tool call
                                    try:
                                        parsed_args = json.loads(tool_call_args_buffer)
                                    except json.JSONDecodeError:
                                        parsed_args = {"raw": tool_call_args_buffer}
                                    yield StreamEvent(
                                        type=EventType.TOOL_USE,
                                        data={
                                            "tool_use_id": current_tool_call["id"],
                                            "tool_name": current_tool_call["name"],
                                            "tool_input": parsed_args,
                                        },
                                    )
                                current_tool_call = {
                                    "id": tc.id or f"call_{id(tc)}",
                                    "name": tc.function.name,
                                }
                                tool_call_args_buffer = tc.function.arguments or ""
                            elif tc.function and tc.function.arguments:
                                # Continuing arguments for the current tool call
                                tool_call_args_buffer += tc.function.arguments

                # Emit the last tool call if any
                if current_tool_call:
                    try:
                        parsed_args = json.loads(tool_call_args_buffer) if tool_call_args_buffer else {}
                    except json.JSONDecodeError:
                        parsed_args = {"raw": tool_call_args_buffer}
                    yield StreamEvent(
                        type=EventType.TOOL_USE,
                        data={
                            "tool_use_id": current_tool_call["id"],
                            "tool_name": current_tool_call["name"],
                            "tool_input": parsed_args,
                        },
                    )

                # Success — exit the retry loop
                return

            except Exception as e:
                last_error = e
                logger.error(f"LLM call failed with {target_model} (attempt {attempt}/{self.config.max_retries + 1}): {e}")

                if not _is_retryable(e) or attempt > self.config.max_retries:
                    # Non-retryable error or retries exhausted — break to fallback
                    break

                # Exponential backoff: 0.5s, 1s, 2s, ...
                delay = (self.config.retry_base_delay_ms / 1000) * (2 ** (attempt - 1))
                logger.info(f"Retrying in {delay:.1f}s (attempt {attempt}/{self.config.max_retries})...")
                yield StreamEvent(
                    type=EventType.STATUS,
                    data={"message": f"API error, retrying in {delay:.0f}s... (attempt {attempt}/{self.config.max_retries})"},
                )
                await asyncio.sleep(delay)

        # All retries exhausted — try fallback model
        if last_error and target_model != self.config.fallback_model:
            logger.info(f"Falling back to {self.config.fallback_model}")
            yield StreamEvent(
                type=EventType.STATUS,
                data={"message": f"Switched to fallback model: {self.config.fallback_model}"},
            )
            async for event in self.stream(
                messages=messages,
                system_prompt=system_prompt,
                tools=tools,
                model=self.config.fallback_model,
                max_tokens=max_tokens,
            ):
                yield event
        elif last_error:
            yield StreamEvent(
                type=EventType.ERROR,
                data={"message": f"LLM call failed: {str(last_error)}"},
            )


async def side_query(
    prompt: str,
    system: str = "",
    model: str | None = None,
    max_tokens: int = 512,
    output_schema: dict | None = None,
    config: ModelConfig | None = None,
) -> str:
    """
    Quick, non-streaming LLM call for routing/ranking/quality checks.

    Uses a cheap, fast model for tasks like:
    - Ranking search results by relevance
    - Selecting relevant memories
    - Quality-checking the final answer

    This is NOT part of the main conversation — it's a separate call
    that doesn't appear in the chat history.

    The model and base_url are resolved from the config if provided,
    otherwise from defaults.
    """
    cfg = config or _shared_config or ModelConfig()
    target_model = model or cfg.side_query_model

    messages = [{"role": "user", "content": prompt}]
    kwargs: dict = {
        "model": target_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "max_completion_tokens": max_tokens,
    }
    if system:
        kwargs["messages"] = [{"role": "system", "content": system}] + messages

    if output_schema:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": output_schema},
        }

    # Apply base_url: prefer side_query_base_url, fall back to base_url
    base_url = cfg.side_query_base_url or cfg.base_url
    if base_url:
        kwargs["api_base"] = base_url

    try:
        response = await litellm.acompletion(**kwargs)
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.warning(f"Side query failed: {e}")
        return ""


# Module-level shared config — set by the router at startup so that
# side_query() (called from compact.py, retrieval.py) picks it up
# without needing the config passed explicitly every time.
_shared_config: ModelConfig | None = None


def set_shared_config(config: ModelConfig) -> None:
    """Set the module-level shared config used by side_query()."""
    global _shared_config
    _shared_config = config
