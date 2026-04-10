"""
Memory type taxonomy.

Mirrors Claude Code's memoryTypes.ts — defines what kinds of memories
the system can store and retrieve. Each type has different persistence
and relevance characteristics.
"""

from __future__ import annotations

from enum import Enum


class MemoryType(str, Enum):
    """Types of persistent memories the system stores."""

    # User profile: who the user is, their background, expertise level
    USER = "user"

    # Feedback: corrections on search behavior, source preferences,
    # quality standards the user has expressed
    FEEDBACK = "feedback"

    # Source reputation: trusted/untrusted sources, domains to prefer/avoid
    SOURCE_REPUTATION = "source_reputation"

    # Reference: bookmarked sources, preferred databases, useful URLs
    REFERENCE = "reference"
