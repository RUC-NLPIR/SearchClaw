"""Local read tool — read specific files/sections under granted roots.

The CLI counterpart to deep_read: instead of reading cached web content, it
reads the user's local files (granted via ``@path``). Supports whole-file
reads, explicit line ranges, and heading/keyword section extraction.

Security mirrors deep_read but generalizes the single cache directory to the
set of user-granted roots (see local_common.resolve_within_roots).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from src.core.tool import Tool, ToolUseContext
from src.core.types import ToolResult, ValidationResult
from src.tools.doc_extract import DegradeNotice, extract_text, is_extractable
from src.tools.local_common import (
    is_skippable_file,
    looks_binary,
    relpath_to_roots,
    resolve_within_roots,
)

logger = logging.getLogger(__name__)

_NO_ROOTS_MSG = (
    "No local directories have been granted. The user must point at a "
    "directory or file with an @path mention before local files can be read."
)


class LocalReadTool(Tool):
    name = "local_read"
    description = (
        "Read a local file the user has granted (via @path). Reads plain-text "
        "files and also extracts text from PDF, .docx, and .pptx documents. "
        "Read the whole file, a specific line range, or a section matching a "
        "heading/keyword. Use after local_search to read context around a match."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file (must be inside a granted root).",
            },
            "section_query": {
                "type": "string",
                "description": (
                    "Optional: a heading name or keyword to extract just the "
                    "relevant section."
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
        "required": ["path"],
    }

    is_concurrency_safe = True
    is_read_only = True

    def __init__(self, max_result_size_chars: int = 30000):
        self.max_result_size_chars = max_result_size_chars

    def prompt(self) -> str:
        return (
            "Use local_read to read the user's granted local files. Provide a "
            "path from a local_search hit, optionally with a line range or a "
            "section_query to target the relevant part."
        )

    def validate_input(self, args: dict) -> ValidationResult:
        if not args.get("path"):
            return ValidationResult(valid=False, message="path is required")
        return ValidationResult(valid=True)

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        roots = list(context.extra.get("allowed_roots") or [])
        if not roots:
            return ToolResult(data=_NO_ROOTS_MSG, is_error=True)

        path_arg = args["path"]
        section_query = args.get("section_query", "")
        start_line = args.get("start_line")
        end_line = args.get("end_line")

        resolved = resolve_within_roots(path_arg, roots)
        if resolved is None:
            logger.warning(f"local_read blocked: '{path_arg}' outside granted roots")
            return ToolResult(
                data=(
                    f"Access denied: '{path_arg}' is outside the granted roots. "
                    f"Only files under these directories can be read: "
                    f"{', '.join(roots)}"
                ),
                is_error=True,
            )
        if not resolved.is_file():
            return ToolResult(data=f"Not a file: {path_arg}", is_error=True)
        if is_skippable_file(resolved):
            return ToolResult(
                data=f"Cannot read '{path_arg}': secret or too large.",
                is_error=True,
            )

        # Binary documents (PDF/docx/pptx) are extracted to text; everything
        # else is read as text after a binary sniff. The extracted text then
        # flows through the same line-range / section logic below.
        if is_extractable(resolved):
            extracted = extract_text(resolved)
            if isinstance(extracted, DegradeNotice):
                return ToolResult(data=str(extracted), is_error=True)
            content = extracted or ""
        else:
            if looks_binary(resolved):
                return ToolResult(
                    data=f"Cannot read '{path_arg}': appears to be binary.",
                    is_error=True,
                )
            try:
                content = resolved.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                return ToolResult(data=f"Failed to read file: {e}", is_error=True)

        rel = relpath_to_roots(resolved, roots)
        total_lines = content.count("\n") + 1

        if start_line is not None or end_line is not None:
            lines = content.splitlines()
            start = max(0, (start_line or 1) - 1)
            end = min(len(lines), end_line or len(lines))
            section = "\n".join(lines[start:end])
            return ToolResult(
                data=f"## {rel} — lines {start + 1}-{end} of {total_lines}\n\n{section}",
            )

        if section_query:
            section = self._extract_section(content, section_query)
            if section:
                return ToolResult(
                    data=f"## {rel} — section matching '{section_query}'\n\n{section}",
                )

        preview = content[: self.max_result_size_chars]
        if len(content) > self.max_result_size_chars:
            preview += (
                f"\n\n---\n[Showing first {self.max_result_size_chars:,} of "
                f"{len(content):,} chars from {rel}. Use start_line/end_line "
                f"for specific ranges. Total lines: {total_lines}]"
            )
        return ToolResult(data=f"## {rel}\n\n{preview}")

    def _extract_section(self, content: str, query: str) -> str | None:
        """Extract a section by heading match, else paragraphs with keywords."""
        query_lower = query.lower()
        lines = content.splitlines()
        heading_pattern = re.compile(r"^#{1,6}\s+(.+)$")

        for i, line in enumerate(lines):
            match = heading_pattern.match(line)
            if match and query_lower in match.group(1).lower():
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

        keywords = query_lower.split()
        if keywords:
            paragraphs = content.split("\n\n")
            matching = [p for p in paragraphs if any(kw in p.lower() for kw in keywords)]
            if matching:
                result = "\n\n".join(matching)
                if len(result) > self.max_result_size_chars:
                    result = result[: self.max_result_size_chars] + "\n\n[Content truncated]"
                return result
        return None
