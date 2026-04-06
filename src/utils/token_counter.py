"""
Token estimation utilities.

Uses tiktoken for fast, accurate token counting. Falls back to
a word-based heuristic if tiktoken is unavailable.

Used by the compaction system to decide when to compact.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Lazy-loaded tiktoken encoder
_encoder = None


def _get_encoder():
    """Lazy-load the tiktoken encoder."""
    global _encoder
    if _encoder is None:
        try:
            import tiktoken
            _encoder = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"tiktoken not available, using heuristic: {e}")
            _encoder = "heuristic"
    return _encoder


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text string.

    Uses tiktoken with cl100k_base encoding (used by GPT-4, Claude, etc.)
    for accuracy. Falls back to word-count heuristic (~1.3 tokens/word)
    if tiktoken is unavailable.
    """
    if not text:
        return 0

    encoder = _get_encoder()

    if encoder == "heuristic":
        # Rough heuristic: ~4 chars per token on average
        return len(text) // 4

    try:
        return len(encoder.encode(text))
    except Exception:
        # Fallback for any encoding error
        return len(text) // 4
