"""Local glob tool — list files/directories under granted roots by name.

The CLI's "ls / find" counterpart to local_search. Where local_search greps
*content* (and so is blind to PDFs and to files whose relevance is in their
name), local_glob matches *filenames* — so it answers "what is in this
directory" and "is there a file called X" even for binary documents.

This is the piece that lets the agent behave like a coding agent: explore a
directory first (local_glob), then grep content (local_search), then read
(local_read). Security mirrors the other local tools — every listed path is
validated to sit inside a user-granted root (see local_common).
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.core.tool import Tool, ToolUseContext
from src.core.types import ToolResult, ValidationResult
from src.tools.local_common import (
    is_skippable_dir,
    relpath_to_roots,
    resolve_within_roots,
)

logger = logging.getLogger(__name__)

_NO_ROOTS_MSG = (
    "No local directories have been granted. The user must point at a "
    "directory or file with an @path mention (e.g. `@~/docs`) before local "
    "files can be listed."
)


def _fmt_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(size)}B"
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


class LocalGlobTool(Tool):
    name = "local_glob"
    description = (
        "List files and directories the user has granted via @path, matching "
        "a filename glob pattern (default '*' = everything). Matches by NAME, "
        "so it finds PDFs, Word/PowerPoint, and other documents that "
        "local_search (grep) cannot see. Use this FIRST to discover what is in "
        "a directory or to check whether a file exists, then local_read to "
        "open a file or local_search to grep text content."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": (
                    "Filename glob, e.g. '*', '*.pdf', '*driving*'. Use '**/*' "
                    "or set recursive=true to descend into subdirectories. "
                    "Defaults to '*' (top level of the listed path)."
                ),
            },
            "path": {
                "type": "string",
                "description": (
                    "Optional: directory to list (must be inside a granted "
                    "root). Defaults to all granted roots."
                ),
            },
            "recursive": {
                "type": "boolean",
                "description": (
                    "If true, search subdirectories too (equivalent to "
                    "prefixing the pattern with '**/'). Default false."
                ),
                "default": False,
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum entries to return (default 200).",
                "default": 200,
            },
        },
        "required": [],
    }

    is_concurrency_safe = True
    is_read_only = True

    def __init__(self, max_results: int = 200):
        self.default_max_results = max_results

    def prompt(self) -> str:
        return (
            "Use local_glob to explore the user's granted directories by "
            "filename (it sees PDFs and other documents that grep cannot). "
            "List a directory before searching or reading it; follow up with "
            "local_read to open a file or local_search to grep text."
        )

    def validate_input(self, args: dict) -> ValidationResult:
        return ValidationResult(valid=True)

    def _roots(self, context: ToolUseContext) -> list[str]:
        return list(context.extra.get("allowed_roots") or [])

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        roots = self._roots(context)
        if not roots:
            return ToolResult(data=_NO_ROOTS_MSG, is_error=True)

        pattern = (args.get("pattern") or "*").strip() or "*"
        recursive = bool(args.get("recursive"))
        max_results = int(args.get("max_results") or self.default_max_results)

        # Determine which directories to list.
        path_arg = args.get("path")
        if path_arg:
            target = resolve_within_roots(path_arg, roots)
            if target is None:
                return ToolResult(
                    data=(
                        f"Access denied: '{path_arg}' is outside the granted "
                        f"roots. Listing is limited to: {', '.join(roots)}"
                    ),
                    is_error=True,
                )
            if target.is_file():
                # A file path: just report the file itself.
                return ToolResult(data=self._format([target], roots, pattern))
            search_dirs = [target]
        else:
            search_dirs = [Path(r) for r in roots]

        # '**/' in the pattern or recursive=true means descend.
        if recursive and not pattern.startswith("**"):
            glob_pattern = f"**/{pattern}"
        else:
            glob_pattern = pattern

        entries: list[Path] = []
        seen: set[str] = set()
        truncated = False
        for d in search_dirs:
            if not d.is_dir():
                continue
            try:
                it = d.glob(glob_pattern)
            except (ValueError, OSError) as e:
                logger.warning(f"local_glob bad pattern '{glob_pattern}': {e}")
                return ToolResult(
                    data=f"Invalid glob pattern '{pattern}': {e}", is_error=True
                )
            for p in it:
                # Skip noise/secret directories anywhere in the relative path.
                if self._in_skipped_dir(p, d):
                    continue
                key = str(p)
                if key in seen:
                    continue
                seen.add(key)
                entries.append(p)
                if len(entries) >= max_results:
                    truncated = True
                    break
            if truncated:
                break

        if not entries:
            scope = path_arg or ", ".join(roots)
            return ToolResult(
                data=(
                    f"No files match '{pattern}' under {scope}. "
                    "The directory may be empty, or try a broader pattern "
                    "(e.g. '*') or recursive=true."
                )
            )

        body = self._format(entries, roots, pattern)
        if truncated:
            body += f"\n\n[Truncated at {max_results} entries — narrow the pattern to see more.]"
        return ToolResult(data=body)

    def _in_skipped_dir(self, path: Path, base: Path) -> bool:
        """True if any path component (below base) is a skippable dir."""
        try:
            rel = path.relative_to(base)
        except ValueError:
            return False
        parts = rel.parts
        # Check every component except the final filename.
        for comp in parts[:-1]:
            if is_skippable_dir(comp):
                return True
        # A matched directory that is itself skippable.
        if path.is_dir() and is_skippable_dir(path.name):
            return True
        return False

    def _format(self, entries: list[Path], roots: list[str], pattern: str) -> str:
        """Render entries as a sorted listing: dirs first, then files."""
        dirs: list[str] = []
        files: list[tuple[str, int]] = []
        for p in entries:
            rel = relpath_to_roots(p, roots)
            try:
                if p.is_dir():
                    dirs.append(rel)
                else:
                    files.append((rel, p.stat().st_size))
            except OSError:
                files.append((rel, 0))

        dirs.sort()
        files.sort(key=lambda t: t[0])

        lines: list[str] = []
        for d in dirs:
            lines.append(f"  {d}/")
        for rel, size in files:
            lines.append(f"  {rel}  ({_fmt_size(size)})")

        n_dirs = len(dirs)
        n_files = len(files)
        header = (
            f"Found {n_dirs} dir(s) and {n_files} file(s) matching '{pattern}':\n"
        )
        return header + "\n".join(lines)
