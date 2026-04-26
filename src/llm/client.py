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
    if "timeout" in error_str or "timeout" in error_type.lower() or "408" in error_str:
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
    reasoning_effort: str = ""  # "minimal", "low", "medium", "high", "xhigh", or "" for default

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
                reasoning_effort=llm.get("reasoning_effort", "") or "",
            )
        except Exception as e:
            logger.warning(f"Failed to load settings from {path}: {e}, using defaults")
            return cls()


def _is_gpt_model(model: str) -> bool:
    """
    Check if a model is an OpenAI GPT/reasoning model that should use
    the Responses API instead of Chat Completions.

    Responses API handles reasoning model state (hidden reasoning tokens,
    previous_response_id chaining) correctly, avoiding "Item with id not found"
    errors that occur with Chat Completions.
    """
    lower = model.lower()
    prefixes = (
        "openai/gpt", "gpt-", "gpt5",
        "openai/o1", "openai/o3", "openai/o4",
        "o1-", "o3-", "o4-",
    )
    return any(lower.startswith(p) for p in prefixes)


def _convert_tools_to_responses_format(tools: list[dict]) -> list[dict]:
    """
    Convert tool schemas from Chat Completions format to Responses API format.

    Chat Completions: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
    Responses API:    {"type": "function", "name": ..., "description": ..., "parameters": ...}
    """
    converted = []
    for t in tools:
        if t.get("type") == "function" and "function" in t:
            func = t["function"]
            converted.append({
                "type": "function",
                "name": func["name"],
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
            })
        else:
            # Already in Responses format or unknown — pass through
            converted.append(t)
    return converted


def _extract_delta_items(messages: list[dict]) -> list[dict]:
    """
    Extract items to send as delta on a Responses API continuation call.

    Finds all tool-result messages after the last assistant message
    (these correspond to the function calls from the previous response),
    plus any user messages injected after them (e.g., plan nudge, synthesis).

    Returns a list of function_call_output and user-role items.
    """
    # Find the index of the last assistant message
    last_assistant_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            last_assistant_idx = i
            break

    if last_assistant_idx < 0:
        return []

    items = []
    for msg in messages[last_assistant_idx + 1:]:
        role = msg.get("role", "")
        if role == "tool":
            items.append({
                "type": "function_call_output",
                "call_id": msg.get("tool_call_id", ""),
                "output": msg.get("content", ""),
            })
        elif role == "user":
            items.append({
                "role": "user",
                "content": msg.get("content", ""),
            })

    return items


class LLMClient:
    """
    Wrapper around litellm for streaming LLM calls.

    Handles:
    - Streaming responses with tool calling
    - Model fallback on failure
    """

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        # Per-session response ID tracking for Responses API chaining.
        # Keyed by session_id so concurrent sessions don't interfere.
        self._response_ids: dict[str, str] = {}

    def reset_response_chain(self, session_id: str = "") -> None:
        """Reset the Responses API chain for a session."""
        self._response_ids.pop(session_id, None)

    @property
    def uses_responses_api(self) -> bool:
        """Whether the default model uses the Responses API (GPT/reasoning models)."""
        return _is_gpt_model(self.config.default_model)

    async def stream(
        self,
        messages: list[dict],
        system_prompt: str,
        tools: list[dict] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        session_id: str = "",
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream an LLM response, yielding StreamEvents.

        Retries transient errors (rate limits, overload, connection) with
        exponential backoff, then falls back to fallback model. Permanent
        errors (400 Bad Request) fail immediately.
        """
        target_model = model or self.config.default_model
        target_max_tokens = max_tokens or self.config.max_tokens

        # Dispatch: use Responses API for GPT/reasoning models
        if _is_gpt_model(target_model):
            async for event in self._stream_responses(
                messages=messages,
                system_prompt=system_prompt,
                tools=tools,
                model=target_model,
                max_tokens=target_max_tokens,
                session_id=session_id,
            ):
                yield event
            return

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
                session_id=session_id,
            ):
                yield event
        elif last_error:
            yield StreamEvent(
                type=EventType.ERROR,
                data={"message": f"LLM call failed: {str(last_error)}"},
            )

    async def _stream_responses(
        self,
        messages: list[dict],
        system_prompt: str,
        tools: list[dict] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        session_id: str = "",
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream via litellm Responses API with previous_response_id chaining.

        Used for OpenAI GPT/reasoning models to correctly handle hidden
        reasoning state. On the first call (no previous_response_id),
        sends the system prompt + user query. On subsequent calls, only sends
        the delta (new function_call_output items).

        session_id isolates the response chain so concurrent sessions
        don't overwrite each other's previous_response_id.
        """
        target_model = model or self.config.default_model
        target_max_tokens = max_tokens or self.config.max_tokens

        # Per-session response ID
        prev_response_id = self._response_ids.get(session_id)

        # Convert tools to Responses API format
        api_tools = _convert_tools_to_responses_format(tools) if tools else None

        # Build input items
        if prev_response_id is None:
            # First call — send system prompt + all messages.
            # Convert to Responses API format (developer/user/assistant roles).
            # Do NOT include tool_calls or tool-role messages — those use
            # call IDs from Chat Completions that the Responses API won't
            # recognise.  The caller (_final_answer) strips these before
            # passing clean_messages.
            input_items = []
            if system_prompt:
                input_items.append({"role": "developer", "content": system_prompt})
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role in ("user", "assistant") and content:
                    input_items.append({"role": role, "content": content})
        else:
            # Continuation — only send new tool results (delta)
            input_items = _extract_delta_items(messages)
            # If no tool results but we have a previous response, it might be
            # a forced final answer with a new user message
            if not input_items:
                # Find the last user message (e.g., synthesis request)
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        input_items = [{"role": "user", "content": msg.get("content", "")}]
                        break

        last_error: Exception | None = None

        for attempt in range(1, self.config.max_retries + 2):
            try:
                kwargs: dict = {
                    "model": target_model,
                    "input": input_items,
                    "max_output_tokens": target_max_tokens,
                    "truncation": "auto",
                    "store": True,  # Required for previous_response_id chaining
                }

                if api_tools:
                    kwargs["tools"] = api_tools

                if prev_response_id:
                    kwargs["previous_response_id"] = prev_response_id

                if self.config.reasoning_effort:
                    kwargs["reasoning"] = {"effort": self.config.reasoning_effort}

                base_url = self.config.base_url
                if base_url:
                    kwargs["api_base"] = base_url

                logger.debug(
                    f"Responses API call: model={target_model}, "
                    f"input_items={len(input_items)}, tools={len(api_tools) if api_tools else 0}, "
                    f"prev_id={prev_response_id}"
                )

                response = await asyncio.wait_for(
                    litellm.aresponses(**kwargs),
                    timeout=600,  # 10 min timeout per Responses API call
                )

                # Save response ID for chaining (per-session)
                prev_response_id = response.id
                self._response_ids[session_id] = response.id

                # Log response output types for debugging
                output_summary = []
                for item in response.output:
                    item_type = getattr(item, "type", None)
                    if item_type == "message":
                        texts = [getattr(p, "text", "")[:100] for p in getattr(item, "content", []) if getattr(p, "type", None) == "output_text"]
                        output_summary.append(f"message({texts})")
                    elif item_type == "function_call":
                        output_summary.append(f"function_call({item.name})")
                    else:
                        output_summary.append(f"{item_type}")
                logger.info(f"Responses API response: id={response.id}, output={output_summary}")

                # Parse response.output and yield StreamEvents
                has_text = False
                for item in response.output:
                    item_type = getattr(item, "type", None)

                    if item_type == "message":
                        # Message item contains content parts
                        for part in getattr(item, "content", []):
                            if getattr(part, "type", None) == "output_text":
                                has_text = True
                                yield StreamEvent(
                                    type=EventType.TEXT_DELTA,
                                    data={"text": part.text},
                                )

                    elif item_type == "function_call":
                        # Tool call
                        try:
                            parsed_args = json.loads(item.arguments)
                        except (json.JSONDecodeError, AttributeError):
                            parsed_args = {"raw": getattr(item, "arguments", "")}

                        yield StreamEvent(
                            type=EventType.TOOL_USE,
                            data={
                                "tool_use_id": item.call_id,
                                "tool_name": item.name,
                                "tool_input": parsed_args,
                            },
                        )

                if not has_text and not any(getattr(item, "type", None) == "function_call" for item in response.output):
                    output_types = [getattr(item, "type", "?") for item in response.output]
                    logger.warning(f"Responses API returned no text and no function calls. Output types: {output_types}")

                # Success — exit retry loop
                return

            except Exception as e:
                last_error = e
                logger.error(
                    f"Responses API call failed with {target_model} "
                    f"(attempt {attempt}/{self.config.max_retries + 1}): {e}"
                )

                if not _is_retryable(e) or attempt > self.config.max_retries:
                    break

                delay = (self.config.retry_base_delay_ms / 1000) * (2 ** (attempt - 1))
                logger.info(f"Retrying in {delay:.1f}s (attempt {attempt}/{self.config.max_retries})...")
                yield StreamEvent(
                    type=EventType.STATUS,
                    data={"message": f"API error, retrying in {delay:.0f}s... (attempt {attempt}/{self.config.max_retries})"},
                )
                await asyncio.sleep(delay)

        # All retries exhausted — try fallback model
        if last_error and model != self.config.fallback_model:
            logger.info(f"Falling back to {self.config.fallback_model}")
            yield StreamEvent(
                type=EventType.STATUS,
                data={"message": f"Switched to fallback model: {self.config.fallback_model}"},
            )
            # Reset chain for fallback (different model)
            self._response_ids.pop(session_id, None)
            async for event in self.stream(
                messages=messages,
                system_prompt=system_prompt,
                tools=tools,
                model=self.config.fallback_model,
                max_tokens=max_tokens,
                session_id=session_id,
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
