"""
Core types for the search agent harness.

These types flow through the entire system — the agentic loop, tools,
hooks, and WebSocket events. Mirrors Claude Code's types/message.ts
and Tool.ts type definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal


# ---------------------------------------------------------------------------
# Citations — accumulated across all tool calls in a research session
# ---------------------------------------------------------------------------

class SourceType(str, Enum):
    WEB = "web"
    ACADEMIC = "academic"
    NEWS = "news"


@dataclass
class Citation:
    url: str
    title: str
    snippet: str
    source_type: SourceType = SourceType.WEB
    accessed_at: datetime = field(default_factory=datetime.now)
    relevance_score: float = 0.0  # 0.0–1.0, set by ranking
    cited: bool = False  # True only for explicitly cited sources (via cite_source)

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "source_type": self.source_type.value,
            "accessed_at": self.accessed_at.isoformat(),
            "relevance_score": self.relevance_score,
            "cited": self.cited,
        }


# ---------------------------------------------------------------------------
# Messages — the conversation history flowing through the loop
# ---------------------------------------------------------------------------

@dataclass
class ContentBlock:
    """A single block of content in a message (text, tool_use, tool_result, etc.)."""
    type: Literal["text", "tool_use", "tool_result"]
    # For text blocks
    text: str | None = None
    # For tool_use blocks
    tool_name: str | None = None
    tool_use_id: str | None = None
    tool_input: dict | None = None
    # For tool_result blocks
    content: str | None = None
    is_error: bool = False


@dataclass
class Message:
    """
    A single message in the conversation.

    Unlike Claude Code which uses a rich union type (AssistantMessage | UserMessage | ...),
    we keep it simple with a role + content model. Metadata carries extra info
    (citations, tool_use_id, compaction markers, etc.).
    """
    role: Literal["user", "assistant", "system", "tool"]
    content: str | list[ContentBlock]
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def text_content(self) -> str:
        """Extract plain text from content, whether string or blocks."""
        if isinstance(self.content, str):
            return self.content
        return " ".join(
            block.text or block.content or ""
            for block in self.content
            if block.type in ("text", "tool_result")
        )

    def to_api_dict(self) -> dict:
        """
        Convert to the format expected by litellm/OpenAI API.

        OpenAI-compatible format:
        - assistant messages with tool calls use "tool_calls" array
        - tool result messages use role="tool" with "tool_call_id"
        """
        # Tool result messages (role="tool")
        if self.role == "tool":
            return {
                "role": "tool",
                "content": self.content if isinstance(self.content, str) else self.text_content,
                "tool_call_id": self.metadata.get("tool_call_id", ""),
            }

        if isinstance(self.content, str):
            return {"role": self.role, "content": self.content}

        # Assistant messages with tool calls — OpenAI format
        if self.role == "assistant":
            msg: dict = {"role": "assistant"}
            text_parts = []
            tool_calls = []

            for block in self.content:
                if block.type == "text" and block.text:
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    import json
                    tool_calls.append({
                        "id": block.tool_use_id,
                        "type": "function",
                        "function": {
                            "name": block.tool_name,
                            "arguments": json.dumps(block.tool_input or {}),
                        },
                    })

            if text_parts:
                msg["content"] = " ".join(text_parts)
            else:
                msg["content"] = None

            if tool_calls:
                msg["tool_calls"] = tool_calls

            return msg

        # Default: simple content
        api_content = []
        for block in self.content:
            if block.type == "text" and block.text:
                api_content.append({"type": "text", "text": block.text})
        return {"role": self.role, "content": api_content if api_content else self.text_content}


# ---------------------------------------------------------------------------
# Tool Results — returned by each tool's call() method
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """
    Result of a tool execution.

    Mirrors Claude Code's ToolResult<T> — data is the main output,
    citations are accumulated across the session, and cached_path
    points to the full content when truncated.
    """
    data: str
    citations: list[Citation] = field(default_factory=list)
    truncated: bool = False
    cached_path: str | None = None
    is_error: bool = False
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stream Events — yielded by the agentic loop, consumed by the WebSocket
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    # Streaming text from the LLM
    TEXT_DELTA = "text_delta"
    # LLM requested a tool call
    TOOL_USE = "tool_use"
    # Tool execution completed
    TOOL_RESULT = "tool_result"
    # New citation discovered
    CITATION = "citation"
    # Status update (compacting, etc.)
    STATUS = "status"
    # Error occurred
    ERROR = "error"
    # Research complete
    DONE = "done"
    # Research plan updated
    PLAN_UPDATE = "plan_update"
    # Interactive question for the user (ask_user tool)
    USER_QUESTION = "user_question"


@dataclass
class StreamEvent:
    """
    Events streamed from the agentic loop to the UI via WebSocket.

    Follow Claude Code's StreamEvent type — the loop is an AsyncGenerator
    that yields these, completely decoupled from the presentation layer.
    """
    type: EventType
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"type": self.type.value, "data": self.data}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    valid: bool
    message: str = ""


# ---------------------------------------------------------------------------
# Loop State — mutable state carried between loop iterations
# ---------------------------------------------------------------------------

@dataclass
class ResearchTask:
    """A single sub-task in a research plan."""
    id: str
    title: str
    details: str = ""
    status: Literal["pending", "in_progress", "completed"] = "pending"
    findings: str = ""


@dataclass
class ResearchPlan:
    """
    Structured research plan for complex queries.

    Inspired by Claude Code's TodoWrite — lets the LLM decompose
    complex questions into tracked sub-tasks, work through them
    systematically, and verify completeness before finalizing.
    """
    tasks: list[ResearchTask] = field(default_factory=list)

    def get_task(self, task_id: str) -> ResearchTask | None:
        """Look up a task by ID."""
        return next((t for t in self.tasks if t.id == task_id), None)

    @property
    def completed_count(self) -> int:
        return sum(1 for t in self.tasks if t.status == "completed")

    @property
    def is_complete(self) -> bool:
        return bool(self.tasks) and all(t.status == "completed" for t in self.tasks)

    def summary(self) -> str:
        """Human-readable summary of plan progress."""
        lines = []
        for t in self.tasks:
            icon = {"pending": "○", "in_progress": "◉", "completed": "●"}[t.status]
            lines.append(f"{icon} [{t.id}] {t.title} — {t.status}")
            if t.findings:
                # Show first 150 chars of findings
                lines.append(f"  → {t.findings[:150]}")
        progress = f"{self.completed_count}/{len(self.tasks)} completed"
        lines.append(f"\nProgress: {progress}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize for streaming to the frontend."""
        return {
            "tasks": [
                {
                    "id": t.id,
                    "title": t.title,
                    "details": t.details,
                    "status": t.status,
                    "findings": t.findings,
                }
                for t in self.tasks
            ],
            "completed_count": self.completed_count,
            "total_count": len(self.tasks),
            "is_complete": self.is_complete,
        }


@dataclass
class LoopState:
    """
    Mutable state carried between iterations of the agentic loop.

    Mirrors Claude Code's query.ts State type — the loop destructures this
    at the top of each iteration and creates a new one at each continue site.
    """
    messages: list[Message]
    turn_count: int = 0
    citations: list[Citation] = field(default_factory=list)
    compaction_count: int = 0
    # Structured research plan (set by research_plan tool)
    research_plan: ResearchPlan | None = None

    @property
    def last_assistant_message(self) -> str | None:
        """Get the text of the most recent assistant message."""
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                return msg.text_content
        return None
