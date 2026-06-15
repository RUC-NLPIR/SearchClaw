"""Per-query assembly and session state, shared by the TUI worker.

Holds the presentation-independent parts of running one research turn:
building QueryParams (memory + system prompt + roots), and finalizing a turn
(updating conversation history, persisting the session, firing memory
extraction). The TUI's worker drives query_loop itself and calls into here so
the assembly logic lives in exactly one place.
"""

from __future__ import annotations

import asyncio
import uuid

from src.cli.mentions import roots_note
from src.cli.runtime import Runtime
from src.core.loop import QueryParams
from src.core.types import Message


class CLISession:
    """Per-session state: conversation history, ids, granted local roots."""

    def __init__(self, runtime: Runtime):
        self.runtime = runtime
        self.conversation_history: list[Message] = []
        self.session_turns: list[dict] = []
        self.session_id: str = str(uuid.uuid4())
        self.last_answer: str = ""  # raw markdown of the most recent answer
        # Local-search roots granted via @path mentions (sticky; cleared on
        # /clear). Files specifically pointed at are tracked for the prompt.
        self.allowed_roots: list[str] = []
        self.focus_files: list[str] = []

    def new_chat(self) -> None:
        self.conversation_history.clear()
        self.session_turns.clear()
        self.session_id = str(uuid.uuid4())
        self.last_answer = ""
        self.allowed_roots.clear()
        self.focus_files.clear()

    def load_history(self, turns: list[dict]) -> list[tuple[str, str]]:
        """Rebuild conversation history from a saved session's turns.

        Only the user query and the model's final answer are restored — tool
        calls are intentionally dropped (they were scaffolding for producing
        the answer and aren't needed for the model to continue the thread).
        Starts a fresh session id so new turns save separately, but seeds the
        conversation so the model has the prior context.

        Returns the list of (query, answer) pairs so the caller can echo them.
        """
        self.conversation_history.clear()
        self.session_turns.clear()
        self.session_id = str(uuid.uuid4())
        self.last_answer = ""
        pairs: list[tuple[str, str]] = []
        for t in turns:
            q = str(t.get("query", "")).strip()
            a = str(t.get("final_answer", "")).strip()
            if not q:
                continue
            self.conversation_history.append(Message(role="user", content=q))
            if a:
                self.conversation_history.append(Message(role="assistant", content=a))
            # Seed session_turns too, so a follow-up turn's save_session keeps
            # the restored history instead of overwriting it with just the new
            # turn (the on-disk session would otherwise lose prior turns).
            self.session_turns.append({
                "query": q,
                "final_answer": a,
                "turn_count": int(t.get("turn_count", 0) or 0),
                "num_citations": int(t.get("num_citations", 0) or 0),
            })
            pairs.append((q, a))
        if pairs:
            self.last_answer = pairs[-1][1]
        return pairs


async def build_query_params(sess: CLISession, query: str) -> QueryParams:
    """Assemble QueryParams for one turn: memory + system prompt + roots."""
    rt = sess.runtime

    memory_content = None
    if rt.memory_enabled:
        try:
            from src.memory.retrieval import find_relevant_memories, format_memories_for_prompt
            relevant = await find_relevant_memories(
                query, rt.memory_store, max_memories=rt.max_relevant_memories
            )
            memory_content = format_memories_for_prompt(relevant)
        except Exception:
            pass

    system_prompt = rt.context_builder.build_system_prompt(
        tools=rt.tool_registry.all_tools(),
        memory_content=memory_content,
        extra_context=roots_note(sess.allowed_roots, sess.focus_files),
    )

    return QueryParams(
        query=query,
        system_prompt=system_prompt,
        tool_registry=rt.tool_registry,
        llm_client=rt.llm_client,
        history=list(sess.conversation_history),
        max_turns=rt.max_turns,
        max_search=rt.max_search,
        max_fetch=rt.max_fetch,
        compact_threshold_tokens=rt.compact_threshold_tokens,
        session_id=sess.session_id,
        hook_engine=rt.hook_engine,
        rate_limiter=rt.rate_limiter,
        cache_dir=rt.cache_dir,
        allowed_roots=list(sess.allowed_roots),
    )


def finalize_turn(
    sess: CLISession,
    query: str,
    session_summary: dict | None,
    done_data: dict | None,
    final_messages: list | None,
) -> None:
    """After a turn ends: update history, persist session, extract memories."""
    rt = sess.runtime

    if session_summary and session_summary.get("final_answer"):
        sess.last_answer = session_summary["final_answer"]

    if final_messages is not None:
        sess.conversation_history = [
            Message(role=m["role"], content=m["content"])
            for m in final_messages
            if isinstance(m, dict) and "role" in m and "content" in m
        ]
    elif session_summary and session_summary.get("final_answer"):
        # Only record the turn when it produced an answer. A failed turn (no
        # answer, no final_messages) must NOT append a lone user message —
        # that leaves the history ending on two consecutive user turns, which
        # providers requiring strict user/assistant alternation reject.
        sess.conversation_history.append(Message(role="user", content=query))
        sess.conversation_history.append(
            Message(role="assistant", content=session_summary["final_answer"])
        )

    if session_summary:
        sess.session_turns.append({
            "query": session_summary.get("query", ""),
            "final_answer": session_summary.get("final_answer", ""),
            "turn_count": done_data.get("turn_count", 0) if done_data else 0,
            "num_citations": len(done_data.get("citations", [])) if done_data else 0,
        })
        try:
            from src.utils.session_storage import SessionStorage
            SessionStorage().save_session(sess.session_id, {
                "query": sess.session_turns[0]["query"],
                "turns": sess.session_turns,
                "final_answer": session_summary.get("final_answer", ""),
                "plan_findings": session_summary.get("plan_findings", ""),
                "turn_count": sum(t.get("turn_count", 0) for t in sess.session_turns),
                "num_citations": len(done_data.get("citations", [])) if done_data else 0,
                "citations": done_data.get("citations", []) if done_data else [],
            })
        except Exception:
            pass
        asyncio.create_task(_extract_memories(session_summary, rt))


async def _extract_memories(summary: dict, rt: Runtime) -> None:
    try:
        from src.memory.extract import extract_memories
        await extract_memories(
            query=summary.get("query", ""),
            final_answer=summary.get("final_answer", ""),
            plan_findings=summary.get("plan_findings", ""),
            store=rt.memory_store,
        )
    except Exception:
        pass
