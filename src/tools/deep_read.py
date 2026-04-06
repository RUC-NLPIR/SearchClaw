"""
Deep read tool — reads specific sections from cached page content.

When web_fetch truncates a large page, the full content is cached to disk.
The deep_read tool lets the agent retrieve specific sections from that
cached content, avoiding the need to re-fetch the page. The cached_path
is set by web_fetch, and deep_read uses it to access the full content.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from src.core.tool import Tool, ToolUseContext
from src.core.types import ToolResult, ValidationResult

logger = logging.getLogger(__name__)


class DeepReadTool(Tool):
    name = "deep_read"
    description = (
        "Read a specific section from a previously fetched and cached web page. "
        "Use this when web_fetch indicates content was truncated and provides a "
        "cached_path. Specify a section query to extract the relevant part."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "cached_path": {
                "type": "string",
                "description": "Path to the cached content file (provided by web_fetch when content is truncated).",
            },
            "section_query": {
                "type": "string",
                "description": (
                    "What section to extract. Can be a heading name, keyword, "
                    "or description like 'the methodology section' or 'results table'."
                ),
            },
            "start_line": {
                "type": "integer",
                "description": "Optional: start reading from this line number.",
            },
            "end_line": {
                "type": "integer",
                "description": "Optional: stop reading at this line number.",
            },
        },
        "required": ["cached_path"],
    }

    is_concurrency_safe = True
    is_read_only = True

    def __init__(self, max_result_size_chars: int = 30000):
        self.max_result_size_chars = max_result_size_chars

    def prompt(self) -> str:
        return (
            "Use deep_read when web_fetch says content was truncated. "
            "Provide the cached_path from the web_fetch result and a section_query "
            "describing what part of the page you need."
        )

    def validate_input(self, args: dict) -> ValidationResult:
        path = args.get("cached_path", "")
        if not path:
            return ValidationResult(valid=False, message="cached_path is required")
        return ValidationResult(valid=True)

    def _is_path_within_cache(self, cached_path: str, cache_dir: Path) -> bool:
        """
        Verify that cached_path is within the allowed cache directory.

        Prevents path traversal attacks where the LLM could be tricked
        into reading arbitrary files on the filesystem (e.g., /etc/passwd,
        ~/.ssh/id_rsa, /proc/self/environ).
        """
        try:
            resolved = Path(cached_path).resolve()
            cache_resolved = cache_dir.resolve()
            return str(resolved).startswith(str(cache_resolved) + os.sep) or resolved == cache_resolved
        except (ValueError, OSError):
            return False

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        cached_path = args["cached_path"]
        section_query = args.get("section_query", "")
        start_line = args.get("start_line")
        end_line = args.get("end_line")

        # Security: verify path is within the allowed cache directory
        if not self._is_path_within_cache(cached_path, context.cache_dir):
            logger.warning(
                f"Path traversal blocked: '{cached_path}' is outside "
                f"cache dir '{context.cache_dir}'"
            )
            return ToolResult(
                data=(
                    f"Access denied: the path '{cached_path}' is outside the "
                    f"allowed cache directory. Only files cached by web_fetch "
                    f"can be read with deep_read."
                ),
                is_error=True,
            )

        try:
            content = Path(cached_path).read_text(encoding="utf-8")
        except Exception as e:
            return ToolResult(
                data=f"Failed to read cached file: {str(e)}",
                is_error=True,
            )

        total_lines = content.count("\n") + 1

        # If line range specified, use that
        if start_line is not None or end_line is not None:
            lines = content.splitlines()
            start = max(0, (start_line or 1) - 1)
            end = min(len(lines), end_line or len(lines))
            section = "\n".join(lines[start:end])
            return ToolResult(
                data=(
                    f"## Lines {start + 1}-{end} of {total_lines} from cached content\n\n"
                    f"{section}"
                ),
            )

        # If section query specified, try to find the relevant section
        if section_query:
            section = self._extract_section(content, section_query)
            if section:
                return ToolResult(
                    data=(
                        f"## Extracted section matching '{section_query}'\n\n"
                        f"{section}"
                    ),
                )

        # No specific section — return an overview with the first chunk
        preview = content[: self.max_result_size_chars]
        if len(content) > self.max_result_size_chars:
            preview += (
                f"\n\n---\n[Showing first {self.max_result_size_chars:,} of "
                f"{len(content):,} chars. Use start_line/end_line to read specific ranges. "
                f"Total lines: {total_lines}]"
            )

        return ToolResult(data=preview)

    def _extract_section(self, content: str, query: str) -> str | None:
        """
        Try to extract a section of the document matching the query.

        Strategies:
        1. Find a heading matching the query
        2. Find paragraphs containing the query keywords
        """
        query_lower = query.lower()

        # Strategy 1: Find matching heading and extract until next heading
        lines = content.splitlines()
        heading_pattern = re.compile(r"^#{1,6}\s+(.+)$")

        for i, line in enumerate(lines):
            match = heading_pattern.match(line)
            if match and query_lower in match.group(1).lower():
                # Found matching heading — extract until next same-level heading
                heading_level = len(line) - len(line.lstrip("#"))
                section_lines = [line]

                for j in range(i + 1, len(lines)):
                    next_match = heading_pattern.match(lines[j])
                    if next_match:
                        next_level = len(lines[j]) - len(lines[j].lstrip("#"))
                        if next_level <= heading_level:
                            break
                    section_lines.append(lines[j])

                section = "\n".join(section_lines)
                if len(section) > self.max_result_size_chars:
                    section = section[: self.max_result_size_chars] + "\n\n[Section truncated]"
                return section

        # Strategy 2: Find paragraphs containing query keywords
        keywords = query_lower.split()
        if keywords:
            paragraphs = content.split("\n\n")
            matching = []
            for para in paragraphs:
                para_lower = para.lower()
                if any(kw in para_lower for kw in keywords):
                    matching.append(para)

            if matching:
                result = "\n\n".join(matching)
                if len(result) > self.max_result_size_chars:
                    result = result[: self.max_result_size_chars] + "\n\n[Content truncated]"
                return result

        return None
