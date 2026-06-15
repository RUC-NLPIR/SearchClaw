"""
Citation registration tool.

Allows the agent to explicitly register a citation for the final answer.
This tool is the formal way to "cite a source" — it records the URL, title,
and a relevant snippet that supports a claim in the answer.

While other tools (web_search, web_fetch) automatically create citations
from their results, cite_source lets the agent curate which citations
actually appear in the final answer.
"""

from __future__ import annotations

import logging

from src.core.tool import Tool, ToolUseContext
from src.core.types import Citation, SourceType, ToolResult, ValidationResult

logger = logging.getLogger(__name__)


class CiteSourceTool(Tool):
    name = "cite_source"
    description = (
        "Register a citation for use in your final answer. Use this to formally "
        "cite a source that supports a specific claim. The citation will appear "
        "in the Sources section of the final response."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL of the source being cited.",
            },
            "title": {
                "type": "string",
                "description": "The title of the source (page title, paper title, etc.).",
            },
            "snippet": {
                "type": "string",
                "description": "A relevant quote or summary from the source that supports your claim.",
            },
            "source_type": {
                "type": "string",
                "enum": ["web", "academic", "news", "local"],
                "description": "The type of source (default: 'web'). Use 'local' for files read from the user's granted @path roots.",
                "default": "web",
            },
            "relevance_note": {
                "type": "string",
                "description": "Optional: why this source is relevant to the user's question.",
            },
        },
        "required": ["url", "title", "snippet"],
    }

    is_concurrency_safe = True
    is_read_only = True
    max_result_size_chars = 5000

    def prompt(self) -> str:
        return (
            "Use cite_source to formally register a citation. Do this for every "
            "source you reference in your answer. The citation should include a "
            "relevant snippet that supports the specific claim you're making."
        )

    def validate_input(self, args: dict) -> ValidationResult:
        if not args.get("url"):
            return ValidationResult(valid=False, message="URL is required")
        if not args.get("title"):
            return ValidationResult(valid=False, message="Title is required")
        if not args.get("snippet"):
            return ValidationResult(valid=False, message="Snippet is required")
        return ValidationResult(valid=True)

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        url = args["url"]
        title = args["title"]
        snippet = args["snippet"]
        source_type_str = args.get("source_type", "web")
        relevance_note = args.get("relevance_note", "")

        # Map string to SourceType enum
        source_type_map = {
            "web": SourceType.WEB,
            "academic": SourceType.ACADEMIC,
            "news": SourceType.NEWS,
            "local": SourceType.LOCAL,
        }
        source_type = source_type_map.get(source_type_str, SourceType.WEB)

        citation = Citation(
            url=url,
            title=title,
            snippet=snippet,
            source_type=source_type,
            relevance_score=1.0,  # Explicitly cited = high relevance
            cited=True,  # Mark as explicitly cited for frontend display
        )

        confirmation = f"Citation registered: [{title}]({url})"
        if relevance_note:
            confirmation += f"\nRelevance: {relevance_note}"

        return ToolResult(
            data=confirmation,
            citations=[citation],
        )
