"""Local search tool — grep-style search within user-granted directories.

The CLI counterpart to web_search: instead of querying the web, it searches
files under the roots the user opted into via ``@path`` mentions. Prefers
ripgrep (``rg``) when installed (fast, respects .gitignore); falls back to a
pure-Python ``os.walk`` + line scan otherwise.

Security: the tool refuses entirely when no roots are granted, and every
search path is validated to sit inside a granted root (see local_common).
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path

from src.core.tool import Tool, ToolUseContext
from src.core.types import Citation, SourceType, ToolResult, ValidationResult
from src.tools.local_common import (
    DOC_EXTS,
    is_skippable_dir,
    is_skippable_file,
    looks_binary,
    relpath_to_roots,
    resolve_within_roots,
)

logger = logging.getLogger(__name__)

_NO_ROOTS_MSG = (
    "No local directories have been granted. The user must point at a "
    "directory or file with an @path mention (e.g. `@~/docs`) before local "
    "search is available. Use web search tools instead, or ask the user to "
    "grant a path."
)


class LocalSearchTool(Tool):
    name = "local_search"
    description = (
        "Search the user's local files (granted via @path) for a text pattern, "
        "grep-style. Returns matching lines with file paths and line numbers. "
        "Use this to find where something is mentioned in the user's own "
        "documents, then use local_read to read the surrounding context."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Text or regular expression to search for (case-insensitive).",
            },
            "path": {
                "type": "string",
                "description": (
                    "Optional: limit the search to this directory or file "
                    "(must be inside a granted root). Defaults to all roots."
                ),
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of matching lines to return (default 40).",
                "default": 40,
            },
        },
        "required": ["query"],
    }

    is_concurrency_safe = True
    is_read_only = True

    def __init__(self, max_results: int = 40, context_chars: int = 200):
        self.default_max_results = max_results
        self.context_chars = context_chars

    def prompt(self) -> str:
        return (
            "Use local_search to grep the user's granted local files. Combine "
            "it with web search: local files for the user's own material, web "
            "for everything else. Follow up promising hits with local_read."
        )

    def validate_input(self, args: dict) -> ValidationResult:
        query = args.get("query", "")
        if not query or len(query.strip()) < 2:
            return ValidationResult(valid=False, message="Query must be at least 2 characters")
        return ValidationResult(valid=True)

    def _roots(self, context: ToolUseContext) -> list[str]:
        return list(context.extra.get("allowed_roots") or [])

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        roots = self._roots(context)
        if not roots:
            return ToolResult(data=_NO_ROOTS_MSG, is_error=True)

        query = args["query"]
        max_results = int(args.get("max_results") or self.default_max_results)

        # Determine search targets: a specific (validated) path, or all roots.
        path_arg = args.get("path")
        if path_arg:
            target = resolve_within_roots(path_arg, roots)
            if target is None:
                return ToolResult(
                    data=(
                        f"Access denied: '{path_arg}' is outside the granted "
                        f"roots. Search is limited to: {', '.join(roots)}"
                    ),
                    is_error=True,
                )
            targets = [target]
        else:
            targets = [Path(r) for r in roots]

        if shutil.which("rg"):
            matches = await self._search_ripgrep(query, targets, max_results)
        else:
            matches = self._search_python(query, targets, max_results)

        # Documents (PDF/docx/pptx) aren't grepped — they're binary on disk.
        # Surface their existence so the agent knows to local_read them.
        doc_hint = self._document_hint(targets)

        if not matches:
            body = f"No grep matches for '{query}' in plain-text files under the granted roots."
            return ToolResult(data=body + doc_hint)

        lines_out: list[str] = []
        citations: list[Citation] = []
        for m in matches:
            rel = relpath_to_roots(Path(m["path"]), roots)
            lines_out.append(f"{rel}:{m['line']}: {m['text']}")
            citations.append(Citation(
                url=f"file://{m['path']}#L{m['line']}",
                title=rel,
                snippet=m["text"][:300],
                source_type=SourceType.LOCAL,
            ))

        header = f"Found {len(matches)} local match(es) for '{query}':\n\n"
        return ToolResult(data=header + "\n".join(lines_out) + doc_hint, citations=citations)

    def _document_hint(self, targets: list[Path], max_listed: int = 20) -> str:
        """List PDF/docx/pptx files under the targets that grep can't search.

        Returns a trailing note (or '') so the agent knows these documents
        exist and can read their text with local_read.
        """
        import os

        docs: list[str] = []
        for target in targets:
            if target.is_file():
                if target.suffix.lower() in DOC_EXTS:
                    docs.append(str(target))
                continue
            for dirpath, dirnames, filenames in os.walk(target):
                dirnames[:] = [d for d in dirnames if not is_skippable_dir(d)]
                for fn in filenames:
                    if Path(fn).suffix.lower() in DOC_EXTS:
                        docs.append(os.path.join(dirpath, fn))
                        if len(docs) > max_listed:
                            break
                if len(docs) > max_listed:
                    break
        if not docs:
            return ""
        shown = docs[:max_listed]
        listing = "\n".join(f"- {d}" for d in shown)
        more = f"\n- … and {len(docs) - max_listed} more" if len(docs) > max_listed else ""
        return (
            "\n\n[Note: grep does not search PDF/Word/PowerPoint files. These "
            f"document(s) under the roots may also be relevant — open them with "
            f"local_read to extract their text:\n{listing}{more}]"
        )

    async def _search_ripgrep(
        self, query: str, targets: list[Path], max_results: int
    ) -> list[dict]:
        """Search via ripgrep --json. rg respects .gitignore by default."""
        cmd = [
            "rg", "--json", "--line-number", "--ignore-case", "--max-filesize", "2M",
            "--max-count", str(max_results), "-e", query,
        ]
        cmd += [str(t) for t in targets]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
        except Exception as e:
            logger.warning(f"ripgrep failed, falling back to Python scan: {e}")
            return self._search_python(query, targets, max_results)

        matches: list[dict] = []
        for raw in stdout.decode("utf-8", "replace").splitlines():
            if len(matches) >= max_results:
                break
            try:
                obj = json.loads(raw)
            except ValueError:
                continue
            if obj.get("type") != "match":
                continue
            d = obj["data"]
            text = (d.get("lines", {}).get("text") or "").rstrip("\n")
            matches.append({
                "path": d["path"]["text"],
                "line": d["line_number"],
                "text": text.strip()[:300],
            })
        return matches

    def _search_python(
        self, query: str, targets: list[Path], max_results: int
    ) -> list[dict]:
        """Pure-Python fallback: walk dirs, scan text files for the query."""
        import os
        import re

        try:
            pattern = re.compile(query, re.IGNORECASE)
        except re.error:
            pattern = re.compile(re.escape(query), re.IGNORECASE)

        matches: list[dict] = []

        def scan_file(fp: Path) -> None:
            if is_skippable_file(fp) or looks_binary(fp):
                return
            try:
                with open(fp, "r", encoding="utf-8", errors="replace") as f:
                    for lineno, line in enumerate(f, 1):
                        if pattern.search(line):
                            matches.append({
                                "path": str(fp),
                                "line": lineno,
                                "text": line.strip()[:300],
                            })
                            if len(matches) >= max_results:
                                return
            except OSError:
                return

        for target in targets:
            if len(matches) >= max_results:
                break
            if target.is_file():
                scan_file(target)
                continue
            for dirpath, dirnames, filenames in os.walk(target):
                dirnames[:] = [d for d in dirnames if not is_skippable_dir(d)]
                for fn in filenames:
                    if len(matches) >= max_results:
                        break
                    scan_file(Path(dirpath) / fn)
        return matches
