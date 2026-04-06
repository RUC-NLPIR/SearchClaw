"""
FastAPI routes — REST endpoints and WebSocket handler.

The WebSocket endpoint /ws/search is the main interface. The frontend
connects via WebSocket, sends a query, and receives streaming events
(text deltas, tool activity, citations, status updates).

REST endpoints provide session history and health checks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.context import ContextBuilder
from src.core.loop import QueryParams, query_loop
from src.core.tool import build_default_registry
from src.core.types import EventType, Message
from src.hooks.builtin_hooks import build_default_hooks
from src.hooks.engine import HookEngine
from src.llm.client import LLMClient, ModelConfig, set_shared_config
from src.memory.retrieval import find_relevant_memories, format_memories_for_prompt
from src.memory.store import MemoryStore
from src.utils.rate_limiter import DomainRateLimiter
from src.utils.session_storage import SessionStorage

logger = logging.getLogger(__name__)

# --- Load config from settings.yaml ---
model_config = ModelConfig.from_settings("config/settings.yaml")
set_shared_config(model_config)  # Make available to side_query() globally

# Load remaining config from same settings file
_memory_base_dir = "./memory"
_memory_enabled = True
_max_relevant_memories = 5
_default_max_turns = 20
_compact_threshold_tokens = 80_000
_rate_limit_per_domain = 10
# Tools config
_web_search_default_results = 10
_web_search_max_results = 20
_academic_search_default_results = 5
_academic_search_max_results = 10
_news_search_default_results = 5
_news_search_max_results = 10
_news_search_default_days_back = 7
_news_search_max_days_back = 30
_max_result_size_chars = 50_000
_http_timeout = 30
_jina_timeout = 60
_content_extraction_threshold = 15_000
_cache_dir = "./cache"
# Hooks config
_min_citations = 2
_min_domains = 2
_min_answer_chars = 200
# Server config
_api_key = ""
_cors_origins = ""
_max_query_length = 10_000
# Browser config
_browser_enabled = False
_browser_use_for_search = True
_browser_use_for_fetch = True
_browser_mode = "playwright"
_browser_cdp_port = 9222
_browser_chrome_path = ""
_browser_headless = True
_browser_user_data_dir = ""
_browser_search_engine = "google"
try:
    import yaml
    _settings_path = Path("config/settings.yaml")
    if _settings_path.exists():
        with open(_settings_path) as _f:
            _settings = yaml.safe_load(_f) or {}
        _memory_cfg = _settings.get("memory", {})
        _memory_base_dir = _memory_cfg.get("base_dir", "./memory")
        _memory_enabled = _memory_cfg.get("enabled", True)
        _max_relevant_memories = int(_memory_cfg.get("max_relevant_memories", 5))
        _limits = _settings.get("limits", {})
        _default_max_turns = int(_limits.get("max_turns", 20))
        _compact_threshold_tokens = int(_limits.get("compact_threshold_tokens", 80_000))
        _rate_limit_per_domain = int(_limits.get("rate_limit_per_domain", 10))
        # Tools config
        _tools_cfg = _settings.get("tools", {})
        _web_search_default_results = int(_tools_cfg.get("web_search_default_results", 10))
        _web_search_max_results = int(_tools_cfg.get("web_search_max_results", 20))
        _academic_search_default_results = int(_tools_cfg.get("academic_search_default_results", 5))
        _academic_search_max_results = int(_tools_cfg.get("academic_search_max_results", 10))
        _news_search_default_results = int(_tools_cfg.get("news_search_default_results", 5))
        _news_search_max_results = int(_tools_cfg.get("news_search_max_results", 10))
        _news_search_default_days_back = int(_tools_cfg.get("news_search_default_days_back", 7))
        _news_search_max_days_back = int(_tools_cfg.get("news_search_max_days_back", 30))
        _max_result_size_chars = int(_tools_cfg.get("max_result_size_chars", 50_000))
        _http_timeout = int(_tools_cfg.get("http_timeout", 30))
        _jina_timeout = int(_tools_cfg.get("jina_timeout", 60))
        _content_extraction_threshold = int(_tools_cfg.get("content_extraction_threshold", 15_000))
        _cache_dir = _tools_cfg.get("cache_dir", "./cache")
        # Hooks config
        _hooks_cfg = _settings.get("hooks", {})
        _min_citations = int(_hooks_cfg.get("min_citations", 2))
        _min_domains = int(_hooks_cfg.get("min_domains", 2))
        _min_answer_chars = int(_hooks_cfg.get("min_answer_chars", 200))
        # Server config
        _server_cfg = _settings.get("server", {})
        _api_key = os.environ.get("SEARCH_CLAW_API_KEY", "") or _server_cfg.get("api_key", "")
        _cors_origins = _server_cfg.get("cors_origins", "")
        # Browser config
        _browser_cfg = _settings.get("browser", {})
        _browser_enabled = _browser_cfg.get("enabled", False)
        _browser_use_for_search = _browser_cfg.get("use_for_search", True)
        _browser_use_for_fetch = _browser_cfg.get("use_for_fetch", True)
        _browser_mode = _browser_cfg.get("mode", "playwright")
        _browser_cdp_port = int(_browser_cfg.get("cdp_port", 9222))
        _browser_chrome_path = _browser_cfg.get("chrome_path", "")
        _browser_headless = _browser_cfg.get("headless", True)
        _browser_user_data_dir = _browser_cfg.get("user_data_dir", "")
        _browser_search_engine = _browser_cfg.get("search_engine", "google")
except Exception:
    pass

logger.info(
    f"LLM config: model={model_config.default_model}, "
    f"side_query={model_config.side_query_model}, "
    f"fallback={model_config.fallback_model}"
    + (f", base_url={model_config.base_url}" if model_config.base_url else "")
)

# --- Shared resources (initialized once) ---
tool_registry = build_default_registry(config={
    "web_search_default_results": _web_search_default_results,
    "web_search_max_results": _web_search_max_results,
    "academic_search_default_results": _academic_search_default_results,
    "academic_search_max_results": _academic_search_max_results,
    "news_search_default_results": _news_search_default_results,
    "news_search_max_results": _news_search_max_results,
    "news_search_default_days_back": _news_search_default_days_back,
    "news_search_max_days_back": _news_search_max_days_back,
    "max_result_size_chars": _max_result_size_chars,
    "http_timeout": _http_timeout,
    "jina_timeout": _jina_timeout,
    "content_extraction_threshold": _content_extraction_threshold,
    # Browser integration
    "browser_search_enabled": _browser_enabled and _browser_use_for_search,
    "browser_fetch_enabled": _browser_enabled and _browser_use_for_fetch,
    "browser_mode": _browser_mode,
    "browser_cdp_port": _browser_cdp_port,
    "browser_chrome_path": _browser_chrome_path,
    "browser_headless": _browser_headless,
    "browser_user_data_dir": _browser_user_data_dir,
    "browser_search_engine": _browser_search_engine,
})
llm_client = LLMClient(config=model_config)
context_builder = ContextBuilder()
memory_store = MemoryStore(base_dir=_memory_base_dir)
logger.info(f"Memory store: {memory_store.base_dir}")
session_storage = SessionStorage()
rate_limiter = DomainRateLimiter(max_per_minute=_rate_limit_per_domain)
logger.info(
    f"Limits: max_turns={_default_max_turns}, "
    f"compact_threshold={_compact_threshold_tokens}, "
    f"rate_limit={_rate_limit_per_domain}/min"
)
logger.info(f"Memory: enabled={_memory_enabled}, max_relevant={_max_relevant_memories}")
logger.info(
    f"Tools: http_timeout={_http_timeout}s, jina_timeout={_jina_timeout}s, "
    f"max_result_size={_max_result_size_chars}, "
    f"web_results={_web_search_default_results}/{_web_search_max_results}, "
    f"cache_dir={_cache_dir}, "
    f"extraction_threshold={_content_extraction_threshold}"
)
logger.info(
    f"Hooks: min_citations={_min_citations}, min_domains={_min_domains}, "
    f"min_answer_chars={_min_answer_chars}"
)
if _browser_enabled:
    _browser_features = []
    if _browser_use_for_search:
        _browser_features.append("search")
    if _browser_use_for_fetch:
        _browser_features.append("fetch")
    logger.info(
        f"Browser: enabled for {'+'.join(_browser_features)}, "
        f"mode={_browser_mode}, headless={_browser_headless}, "
        f"search_engine={_browser_search_engine}"
    )
else:
    logger.info("Browser: disabled")

# Hook engine with default quality hooks
hook_engine = HookEngine()
for hook in build_default_hooks(config={
    "min_citations": _min_citations,
    "min_domains": _min_domains,
    "min_answer_chars": _min_answer_chars,
}):
    hook_engine.register_stop_hook(hook)

# Plan completeness hook — ensures all research plan sub-tasks
# are completed before the agent finalizes its answer
from src.hooks.plan_completeness_hook import PlanCompletenessHook
hook_engine.register_stop_hook(PlanCompletenessHook())


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Manage startup/shutdown lifecycle — clean up httpx clients."""
        logger.info("SearchClaw starting up")
        yield
        # Shutdown: close all httpx clients held by tools
        logger.info("Shutting down — closing HTTP clients")
        for tool in tool_registry.all_tools():
            for attr_name in ("_client", "_jina_client"):
                client = getattr(tool, attr_name, None)
                if client is not None and hasattr(client, "aclose"):
                    try:
                        await client.aclose()
                        logger.debug(f"Closed {attr_name} on {tool.name}")
                    except Exception as e:
                        logger.warning(f"Error closing {attr_name} on {tool.name}: {e}")
        # Shutdown: close browser manager (if initialized)
        try:
            from src.utils.browser_manager import BrowserManager
            await BrowserManager.shutdown_instance()
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Error shutting down browser manager: {e}")

    app = FastAPI(
        title="SearchClaw",
        description="A web research agent powered by LLMs",
        version="0.1.0",
        lifespan=lifespan,
    )

    # --- CORS middleware ---
    if _cors_origins:
        origins = [o.strip() for o in _cors_origins.split(",") if o.strip()]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        logger.info(f"CORS enabled for origins: {origins}")
    else:
        # Default: same-origin only (no CORS headers = browser blocks cross-origin)
        logger.info("CORS not configured — same-origin only")

    # --- API key authentication middleware ---
    if _api_key:
        class APIKeyAuthMiddleware(BaseHTTPMiddleware):
            """Require API key for all requests when configured."""
            async def dispatch(self, request: Request, call_next):
                path = request.url.path

                # Always allow: health check, login endpoint, static assets
                # (static files must load for the login page to render)
                if path == "/api/health" or path == "/api/login" or path.startswith("/static"):
                    return await call_next(request)

                # Allow the HTML page to load (login overlay is rendered client-side)
                if path == "/":
                    return await call_next(request)

                # Check Authorization header: "Bearer <key>"
                auth_header = request.headers.get("authorization", "")
                if auth_header.startswith("Bearer ") and auth_header[7:] == _api_key:
                    return await call_next(request)

                # Check query parameter: ?api_key=<key>
                if request.query_params.get("api_key") == _api_key:
                    return await call_next(request)

                return JSONResponse(
                    status_code=401,
                    content={"error": "Invalid or missing API key"},
                )

        app.add_middleware(APIKeyAuthMiddleware)
        logger.info("API key authentication enabled")
    else:
        logger.warning(
            "No API key configured — all endpoints are unauthenticated. "
            "Set 'server.api_key' in settings.yaml or SEARCH_CLAW_API_KEY env var."
        )

    # Mount static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # --- Routes ---

    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Serve the main UI."""
        index_path = static_dir / "index.html"
        if index_path.exists():
            return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
        return HTMLResponse(content="<h1>SearchClaw</h1><p>Static files not found.</p>")

    @app.get("/api/health")
    async def health():
        """Health check endpoint."""
        return {
            "status": "ok",
            "tools": [t.name for t in tool_registry.all_tools()],
            "model": llm_client.config.default_model,
        }

    @app.post("/api/login")
    async def login(request: Request):
        """Verify password. Returns 200 if correct, 401 if wrong or not configured."""
        if not _api_key:
            return {"ok": True}
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"error": "Invalid JSON"})
        if body.get("password") == _api_key:
            return {"ok": True}
        return JSONResponse(status_code=401, content={"error": "Wrong password"})

    @app.get("/api/sessions")
    async def list_sessions():
        """List recent research sessions."""
        return {"sessions": session_storage.list_sessions(limit=20)}

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str):
        """Get a specific session transcript."""
        session = session_storage.load_session(session_id)
        if session is None:
            return JSONResponse(
                content={"error": "Session not found"},
                status_code=404,
            )
        return session

    @app.websocket("/ws/search")
    async def search_websocket(ws: WebSocket):
        """
        WebSocket endpoint for streaming research.

        Protocol:
        1. Client sends: {"query": "...", "options": {...}}
        2. Server streams: {"type": "text_delta|tool_use|citation|status|done", "data": {...}}
        3. Connection stays open for follow-up queries

        Conversation continuity: maintains a per-connection conversation
        history that persists across queries within the same WebSocket
        session. The client can send options.new_chat=true to reset.
        """
        # --- WebSocket authentication ---
        # HTTP auth middleware doesn't cover WebSocket upgrades, so we
        # check the API key here via query parameter.
        if _api_key:
            api_key_param = ws.query_params.get("api_key", "")
            if api_key_param != _api_key:
                await ws.close(code=4001, reason="Invalid or missing API key")
                return

        await ws.accept()
        logger.info("WebSocket connection established")

        # Per-connection conversation history — persists across queries
        # within the same WebSocket session. Mirrors Claude Code's REPL
        # where messages accumulate across user turns.
        conversation_history: list[Message] = []
        session_id = str(uuid.uuid4())

        # Per-connection query rate tracking
        _query_timestamps: list[float] = []
        _max_queries_per_minute = 10

        try:
            while True:
                # Receive query from client
                raw = await ws.receive_text()
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "data": {"message": "Invalid JSON"}})
                    continue

                query = message.get("query", "").strip()
                if not query:
                    await ws.send_json({"type": "error", "data": {"message": "Query is required"}})
                    continue

                # Query length limit — prevent context overflow and abuse
                if len(query) > _max_query_length:
                    await ws.send_json({
                        "type": "error",
                        "data": {"message": f"Query too long ({len(query)} chars). Maximum is {_max_query_length} chars."},
                    })
                    continue

                # Per-connection query rate limiting
                import time as _time
                now_ts = _time.monotonic()
                _query_timestamps[:] = [t for t in _query_timestamps if now_ts - t < 60]
                if len(_query_timestamps) >= _max_queries_per_minute:
                    await ws.send_json({
                        "type": "error",
                        "data": {"message": "Rate limit exceeded. Please wait before sending more queries."},
                    })
                    continue
                _query_timestamps.append(now_ts)

                # Parse options
                options = message.get("options", {})

                # Check for explicit new-chat signal from frontend
                if options.get("new_chat", False):
                    conversation_history.clear()
                    session_id = str(uuid.uuid4())
                    logger.info(f"New chat started [{session_id[:8]}]")

                max_turns = options.get("max_turns", _default_max_turns)

                logger.info(
                    f"Research query [{session_id}] "
                    f"(history={len(conversation_history)} msgs): {query[:100]}"
                )

                # Load relevant memories
                memory_content = None
                if _memory_enabled:
                    try:
                        relevant_memories = await find_relevant_memories(
                            query, memory_store, max_memories=_max_relevant_memories
                        )
                        memory_content = format_memories_for_prompt(relevant_memories)
                        if relevant_memories:
                            logger.info(
                                f"Loaded {len(relevant_memories)} memories into prompt: "
                                + ", ".join(f"[{m.memory_type.value}] {m.title}" for m in relevant_memories)
                            )
                        else:
                            logger.info("No memories loaded (store empty or none relevant)")
                    except Exception as e:
                        logger.warning(f"Memory retrieval failed: {e}")
                else:
                    logger.debug("Memory system disabled")

                # Build system prompt
                system_prompt = context_builder.build_system_prompt(
                    tools=tool_registry.all_tools(),
                    memory_content=memory_content,
                )

                # Run the agentic loop — pass conversation history
                params = QueryParams(
                    query=query,
                    system_prompt=system_prompt,
                    tool_registry=tool_registry,
                    llm_client=llm_client,
                    history=list(conversation_history),  # Copy to avoid mutation
                    max_turns=max_turns,
                    compact_threshold_tokens=_compact_threshold_tokens,
                    session_id=session_id,
                    hook_engine=hook_engine,
                    rate_limiter=rate_limiter,
                    cache_dir=_cache_dir,
                )

                # Stream events to the WebSocket using manual asend()
                # iteration. This enables bidirectional communication:
                # the loop yields events, and we send back user answers
                # for interactive tools (ask_user) via asend().
                session_summary = None
                done_event_data = None
                final_messages = None
                gen = query_loop(params)
                sent_value: str | None = None
                try:
                    while True:
                        event = await gen.asend(sent_value)
                        sent_value = None  # Reset after each send

                        try:
                            await ws.send_json(event.to_dict())
                        except Exception as e:
                            logger.error(f"Failed to send WebSocket message: {e}")
                            break

                        # Handle interactive user question — wait for
                        # the user's answer on the WebSocket, then send
                        # it back into the generator on the next iteration.
                        if event.type == EventType.USER_QUESTION:
                            try:
                                raw_answer = await asyncio.wait_for(
                                    ws.receive_text(), timeout=120,
                                )
                                answer_msg = json.loads(raw_answer)
                                sent_value = answer_msg.get("answer", "")
                                logger.info(
                                    f"User answered question [{session_id}]: "
                                    f"{sent_value[:80]}"
                                )
                            except asyncio.TimeoutError:
                                logger.warning(
                                    f"User question timed out [{session_id}]"
                                )
                                sent_value = ""  # Default — loop uses first option
                            except (WebSocketDisconnect, Exception) as e:
                                logger.warning(
                                    f"Error reading user answer [{session_id}]: {e}"
                                )
                                sent_value = ""

                        # Capture session summary from DONE event
                        elif event.type == EventType.DONE:
                            done_event_data = event.data
                            if "session_summary" in event.data:
                                session_summary = event.data["session_summary"]
                            # Capture condensed messages for conversation continuity
                            if "final_messages" in event.data:
                                final_messages = event.data["final_messages"]

                except StopAsyncIteration:
                    pass

                # Update conversation history for the next query in this session.
                # final_messages contains serialized user+assistant messages from the loop.
                if final_messages is not None:
                    conversation_history = [
                        Message(role=m["role"], content=m["content"])
                        for m in final_messages
                        if isinstance(m, dict) and "role" in m and "content" in m
                    ]
                else:
                    # Fallback: manually append user query and final answer
                    conversation_history.append(Message(role="user", content=query))
                    if session_summary and session_summary.get("final_answer"):
                        conversation_history.append(Message(
                            role="assistant",
                            content=session_summary["final_answer"],
                        ))

                # Save session transcript
                if session_summary and done_event_data:
                    try:
                        session_storage.save_session(session_id, {
                            **session_summary,
                            "turn_count": done_event_data.get("turn_count", 0),
                            "compaction_count": done_event_data.get("compaction_count", 0),
                            "num_citations": len(done_event_data.get("citations", [])),
                            "citations": done_event_data.get("citations", []),
                        })
                    except Exception as e:
                        logger.warning(f"Failed to save session: {e}")

                # Fire-and-forget: extract memories from this session
                if session_summary:
                    asyncio.create_task(
                        _extract_session_memories(session_summary, memory_store)
                    )

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}", exc_info=True)
            try:
                await ws.send_json({"type": "error", "data": {"message": str(e)}})
            except Exception:
                pass

    async def _extract_session_memories(summary: dict, store: MemoryStore) -> None:
        """Fire-and-forget wrapper for post-session memory extraction."""
        try:
            from src.memory.extract import extract_memories
            saved = await extract_memories(
                query=summary.get("query", ""),
                final_answer=summary.get("final_answer", ""),
                plan_findings=summary.get("plan_findings", ""),
                store=store,
            )
            if saved:
                logger.info(f"Post-session: saved {len(saved)} memories")
        except Exception as e:
            logger.warning(f"Post-session memory extraction failed: {e}")

    return app
