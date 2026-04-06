"""
Hook execution engine.

Runs quality checks at key lifecycle points. Hooks act as quality
gates that can force the agent to continue researching if the answer
doesn't meet standards.

Hook events:
- stop: Run before finalizing an answer (quality gate)
- post_tool: Run after each tool execution (source verification)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.core.types import LoopState

logger = logging.getLogger(__name__)


@dataclass
class HookEvaluation:
    """Result of a single hook evaluation."""
    passed: bool
    feedback: str = ""


@dataclass
class HookResult:
    """Aggregate result from running all hooks at an event."""
    should_continue: bool = False
    feedback: str | None = None


class Hook(ABC):
    """Base class for all hooks."""
    name: str = ""
    description: str = ""

    @abstractmethod
    async def evaluate(self, state: LoopState, **kwargs) -> HookEvaluation:
        """Evaluate this hook against the current state."""
        ...


class HookEngine:
    """
    Runs quality checks at key lifecycle points.

    Stop hooks run when the model wants to finalize, post-tool hooks
    run after each tool call.
    """

    def __init__(self):
        self.stop_hooks: list[Hook] = []

    def register_stop_hook(self, hook: Hook) -> None:
        """Register a hook that runs before finalizing an answer."""
        self.stop_hooks.append(hook)
        logger.info(f"Registered stop hook: {hook.name}")

    async def run_stop_hooks(self, state: LoopState) -> HookResult:
        """
        Run all stop hooks (quality gates before finalizing).

        If any hook fails, returns should_continue=True with feedback
        that will be injected into the conversation to force more research.
        """
        result = HookResult()

        try:
            for hook in self.stop_hooks:
                try:
                    evaluation = await hook.evaluate(state)

                    if not evaluation.passed:
                        logger.info(f"Stop hook '{hook.name}' failed: {evaluation.feedback}")
                        result.should_continue = True
                        result.feedback = evaluation.feedback
                        return result  # First failure stops evaluation

                except Exception as e:
                    logger.warning(f"Stop hook '{hook.name}' error (skipping): {e}")
        finally:
            # Always clear per-cycle caches (e.g., _needs_research cache)
            # to prevent unbounded memory growth across sessions.
            try:
                from src.hooks.builtin_hooks import clear_research_cache
                clear_research_cache()
            except ImportError:
                pass

        return result
