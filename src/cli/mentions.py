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

import os
import re
from pathlib import Path

# @ followed by a quoted path OR a run of non-whitespace chars.
_MENTION_RE = re.compile(r'@(?:"([^"]+)"|\'([^\']+)\'|(\S+))')


def _is_cjk(ch: str) -> bool:
    """True for CJK/kana/hangul/fullwidth characters.

    Used to find where a path ends when no space separates it from the
    following text — common in Chinese/Japanese/Korean input, where a user
    writes ``@/path/目录里有什么`` with the query glued onto the path.
    """
    o = ord(ch)
    return (
        0x3000 <= o <= 0x303F  # CJK symbols & punctuation
        or 0x3040 <= o <= 0x30FF  # hiragana + katakana
        or 0x3400 <= o <= 0x4DBF  # CJK ext A
        or 0x4E00 <= o <= 0x9FFF  # CJK unified ideographs
        or 0xAC00 <= o <= 0xD7AF  # hangul syllables
        or 0xF900 <= o <= 0xFAFF  # CJK compatibility ideographs
        or 0xFF00 <= o <= 0xFFEF  # fullwidth & halfwidth forms
    )


def _longest_existing_prefix(raw: str) -> tuple[Path, int] | None:
    """Longest prefix of ``raw`` that exists on disk, split at a CJK boundary.

    When ``@/tmp/目录里的内容`` is typed without a space, the whole run
    ``/tmp/目录里的内容`` is captured as one token and fails to resolve. Here we
    walk back to the longest prefix that (a) exists and (b) is immediately
    followed by a CJK character — i.e. the point where the path plausibly ends
    and Chinese query text begins. We only break at CJK boundaries so ASCII
    paths (which can't be disambiguated without a space anyway) are untouched,
    and we never accept a filesystem root, which would grant far too much.

    Returns ``(resolved_path, split_index)`` where ``raw[:split_index]`` is the
    path text and ``raw[split_index:]`` is the trailing query, or ``None``.
    """
    for end in range(len(raw) - 1, 0, -1):
        if not _is_cjk(raw[end]):
            continue
        # Don't split right after a separator: a trailing "/" means the
        # component the user actually typed (the CJK run) doesn't exist, so
        # granting its parent would over-grant (e.g. @/tmp/缺失目录 → /tmp).
        if raw[end - 1] in (os.sep, "/"):
            continue
        try:
            p = Path(raw[:end]).expanduser().resolve()
            if not p.exists() or p.parent == p:
                continue
        except (OSError, RuntimeError, ValueError):
            continue
        return p, end
    return None


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
        quoted = match.group(1) is not None or match.group(2) is not None
        trailing = ""
        try:
            p = Path(raw).expanduser().resolve()
        except (OSError, RuntimeError):
            warnings.append(f"@{raw}: invalid path — ignored")
            return ""
        if not p.exists():
            # Unquoted CJK input often glues the query onto the path with no
            # space (e.g. @/path/目录里有什么). Recover the longest real path
            # prefix and hand the rest back to the query text.
            split = None if quoted else _longest_existing_prefix(raw)
            if split is None:
                warnings.append(f"@{raw}: path not found — ignored")
                return ""
            p, idx = split
            trailing = raw[idx:]
        if p.is_dir():
            _add(new_roots, str(p))
        else:
            _add(new_roots, str(p.parent))
            _add(focus_files, str(p))
        # Keep the path text (without the @) in the query so the sentence
        # still reads naturally, e.g. "check the papers in ~/papers" rather
        # than leaving a blank gap where the mention was. When we split a
        # glued CJK token, re-insert a space so the path and the recovered
        # query text don't run together.
        return raw if not trailing else (raw[: len(raw) - len(trailing)] + " " + trailing)

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
        "Three tools work on these roots:\n"
        "- `local_glob`: list files/dirs by NAME (e.g. '*', '*.pdf', "
        "'*driving*'). It sees PDFs and other documents that grep cannot. Use "
        "it FIRST to discover what is in a directory or whether a file exists.\n"
        "- `local_search`: grep file CONTENT for a text pattern (plain-text "
        "files only — not PDFs/Office docs).\n"
        "- `local_read`: read a specific file or line range; also extracts "
        "text from PDF/.docx/.pptx.\n\n"
        "For a question like 'what papers are in this folder', start with "
        "`local_glob` to list the files, then `local_read` the relevant ones — "
        "do NOT rely on `local_search` alone, since most papers are PDFs that "
        "grep cannot read. Prefer local sources when the question concerns the "
        "user's own files, and combine them with web sources where useful. "
        "Cite a local finding with `cite_source` using a `file://` URL of the "
        "form `file:///abs/path#L42` and source_type `local`."
    )
    if focus_files:
        focus_list = "\n".join(f"- {f}" for f in focus_files)
        note += (
            "\n\nThe user specifically pointed at these files — start with "
            f"them:\n{focus_list}"
        )
    return note
