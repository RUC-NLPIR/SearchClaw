"""Parse ``@path`` mentions from CLI input.

Like Claude Code / Codex, the user can prefix a path with ``@`` to grant the
agent local-search access to that directory (or file). Parsing happens in the
CLI input layer — the web path never sees it — so the agent core stays
unchanged except for receiving an ``allowed_roots`` list.

Grammar (lightweight, not a shell): a mention is ``@`` immediately followed by
either a quoted path (``@"with spaces"`` / ``@'…'``) or a run of non-space
characters. Everything else is the query the model sees, with the ``@token``
removed.
"""

from __future__ import annotations

import re
from pathlib import Path

# @ followed by a quoted path OR a run of non-whitespace chars.
_MENTION_RE = re.compile(r'@(?:"([^"]+)"|\'([^\']+)\'|(\S+))')


def parse_mentions(text: str) -> tuple[str, list[str], list[str], list[str]]:
    """Extract ``@path`` mentions from input.

    Returns ``(cleaned_query, new_roots, focus_files, warnings)``:
      - ``cleaned_query``: input with every ``@token`` removed (what the model sees)
      - ``new_roots``: resolved absolute directories to grant (dirs as-is; a
        file's parent directory)
      - ``focus_files``: resolved absolute paths of file mentions, so the prompt
        can point the agent at them specifically
      - ``warnings``: human-readable notes for mentions that didn't resolve

    Non-existent paths are dropped with a warning; the rest of the query is
    preserved. Order is preserved and duplicates are de-duped.
    """
    new_roots: list[str] = []
    focus_files: list[str] = []
    warnings: list[str] = []

    def _add(lst: list[str], item: str) -> None:
        if item not in lst:
            lst.append(item)

    def _replace(match: re.Match) -> str:
        raw = match.group(1) or match.group(2) or match.group(3) or ""
        try:
            p = Path(raw).expanduser().resolve()
        except (OSError, RuntimeError):
            warnings.append(f"@{raw}: invalid path — ignored")
            return ""
        if not p.exists():
            warnings.append(f"@{raw}: path not found — ignored")
            return ""
        if p.is_dir():
            _add(new_roots, str(p))
        else:
            _add(new_roots, str(p.parent))
            _add(focus_files, str(p))
        # Keep the path text (without the @) in the query so the sentence
        # still reads naturally, e.g. "check the papers in ~/papers" rather
        # than leaving a blank gap where the mention was.
        return raw

    cleaned = _MENTION_RE.sub(_replace, text)
    # Collapse only the gaps left by *removed* (invalid) mentions.
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned, new_roots, focus_files, warnings


def roots_note(roots: list[str], focus_files: list[str]) -> str:
    """Build the system-prompt block describing available local roots.

    Returned as a section to pass via ``build_system_prompt(extra_context=…)``.
    Empty string when there are no roots (so the prompt is unchanged).
    """
    if not roots:
        return ""
    root_list = "\n".join(f"- {r}" for r in roots)
    note = (
        "## Local Files\n\n"
        "The user has granted you read-only access to local files under these "
        "roots:\n"
        f"{root_list}\n\n"
        "Use `local_search` to find content within them (grep-style) and "
        "`local_read` to read specific files or line ranges. Prefer local "
        "sources when the question concerns the user's own files, and combine "
        "them with web sources where useful. Cite a local finding with "
        "`cite_source` using a `file://` URL of the form "
        "`file:///abs/path#L42` and source_type `local`."
    )
    if focus_files:
        focus_list = "\n".join(f"- {f}" for f in focus_files)
        note += (
            "\n\nThe user specifically pointed at these files — start with "
            f"them:\n{focus_list}"
        )
    return note
