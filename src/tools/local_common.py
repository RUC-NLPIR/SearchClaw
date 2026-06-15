"""Shared helpers for local-filesystem tools (local_search, local_read).

Centralizes the security boundary: every path the model supplies must resolve
inside one of the user-granted roots (from ``@path`` mentions). Also holds the
default-deny filters (noise dirs, secret files, binaries) and size caps that
keep a stray ``@~`` from flooding the context or leaking credentials.
"""

from __future__ import annotations

import os
from pathlib import Path

# Directories never descended into (noise + VCS internals).
SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".idea", ".vscode",
    "dist", "build", ".next", ".cache", "site-packages", ".gradle",
}

# Filename prefixes/exact names treated as secrets and skipped.
SECRET_NAMES = {
    ".env", ".netrc", ".pgpass", "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519",
    "credentials", ".htpasswd",
}
SECRET_DIR_NAMES = {".ssh", ".aws", ".gnupg", ".kube", ".docker"}

# Per-file and aggregate caps (chars / bytes) to bound context cost.
MAX_FILE_BYTES = 2 * 1024 * 1024  # 2 MB: skip plain-text files larger than this
# Documents (PDF/Office) get a higher cap — a long PDF easily exceeds 2 MB but
# extracts to a reasonable amount of text. local_read routes these through
# doc_extract rather than read_text.
MAX_DOC_BYTES = 30 * 1024 * 1024  # 30 MB

# Binary document formats local_read can extract text from (via doc_extract).
# Kept OUT of BINARY_EXTS so they aren't skipped as opaque binaries.
DOC_EXTS = {".pdf", ".docx", ".pptx"}

# Extensions that are obviously binary; skipped before any content read.
# Note: PDF/docx/pptx are intentionally NOT here — see DOC_EXTS.
BINARY_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".svg",
    ".zip", ".gz", ".tar", ".bz2", ".xz", ".7z", ".rar",
    ".mp3", ".mp4", ".avi", ".mov", ".mkv", ".wav", ".flac", ".ogg",
    ".so", ".dll", ".dylib", ".o", ".a", ".class", ".pyc", ".pyd",
    ".bin", ".exe", ".wasm", ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".parquet", ".npy", ".npz", ".pkl", ".pt", ".onnx", ".h5", ".db", ".sqlite",
}


def resolve_within_roots(path: str, roots: list[str]) -> Path | None:
    """Resolve ``path`` and return it only if it sits inside some root.

    Returns the resolved ``Path`` when safe, else ``None``. ``resolve()``
    collapses ``..`` and follows symlinks, so this blocks traversal and
    symlink escapes. A bare relative path is resolved against each root in
    turn (so the agent can pass ``subdir/file`` without the absolute prefix).
    """
    if not roots:
        return None
    resolved_roots = []
    for r in roots:
        try:
            resolved_roots.append(Path(r).resolve())
        except (OSError, RuntimeError):
            continue

    candidates: list[Path] = []
    p = Path(path).expanduser()
    if p.is_absolute():
        try:
            candidates.append(p.resolve())
        except (OSError, RuntimeError):
            return None
    else:
        # Try the path as-is (relative to cwd) and relative to each root.
        for root in resolved_roots:
            try:
                candidates.append((root / p).resolve())
            except (OSError, RuntimeError):
                continue
        try:
            candidates.append(p.resolve())
        except (OSError, RuntimeError):
            pass

    for cand in candidates:
        for root in resolved_roots:
            if cand == root or str(cand).startswith(str(root) + os.sep):
                return cand
    return None


def is_skippable_dir(name: str) -> bool:
    return name in SKIP_DIRS or name in SECRET_DIR_NAMES


def is_skippable_file(path: Path) -> bool:
    """True if a file should be skipped (secret, binary, or too large).

    Extractable documents (PDF/Office) are allowed a larger size budget than
    plain-text files, since a long PDF is big on disk but yields modest text.
    """
    name = path.name
    if name in SECRET_NAMES:
        return True
    if path.suffix.lower() in BINARY_EXTS:
        return True
    size_cap = MAX_DOC_BYTES if path.suffix.lower() in DOC_EXTS else MAX_FILE_BYTES
    try:
        if path.stat().st_size > size_cap:
            return True
    except OSError:
        return True
    return False


def looks_binary(path: Path, sniff_bytes: int = 4096) -> bool:
    """Cheap binary sniff: a NUL byte in the first chunk means binary."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(sniff_bytes)
        return b"\x00" in chunk
    except OSError:
        return True


def relpath_to_roots(path: Path, roots: list[str]) -> str:
    """Display path relative to its containing root, falling back to abs."""
    for r in roots:
        try:
            root = Path(r).resolve()
            return str(path.relative_to(root))
        except (ValueError, OSError, RuntimeError):
            continue
    return str(path)
