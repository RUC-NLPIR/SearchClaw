"""
Plan completeness stop hook.

Ensures the agent doesn't finalize its answer while there are
still incomplete sub-tasks in the research plan. This is the key
mechanism that prevents the LLM from stopping early on complex
multi-part queries.

If no research plan exists, the hook passes (no constraint).
If all tasks are completed, the hook passes.
Otherwise, the hook fails with feedback listing the remaining tasks.
"""

from __future__ import annotations

import logging

from src.core.types import LoopState
from src.hooks.engine import Hook, HookEvaluation

logger = logging.getLogger(__name__)


class PlanCompletenessHook(Hook):
    """Stop hook that enforces research plan completion."""

    name = "plan_completeness"
    description = "Ensures all research plan sub-tasks are completed before finalizing"

    async def evaluate(self, state: LoopState, **kwargs) -> HookEvaluation:
        # No plan → no constraint
        if state.research_plan is None:
            return HookEvaluation(passed=True)

        plan = state.research_plan

        # Empty plan (shouldn't happen, but be safe)
        if not plan.tasks:
            return HookEvaluation(passed=True)

        # All tasks completed → pass
        if plan.is_complete:
            return HookEvaluation(passed=True)

        # Incomplete tasks → fail with feedback
        pending = [t for t in plan.tasks if t.status != "completed"]
        task_list = "\n".join(f"- [{t.id}] {t.title}" for t in pending)

        feedback = (
            f"Your research plan has {len(pending)} incomplete sub-task(s):\n"
            f"{task_list}\n\n"
            f"Please continue researching these topics before providing your "
            f"final answer. Use web_search and web_fetch to investigate each "
            f"remaining sub-task, then call research_plan(action='update') to "
            f"record your findings."
        )

        logger.info(
            f"Plan completeness check failed: {len(pending)}/{len(plan.tasks)} tasks remaining"
        )

        return HookEvaluation(
            passed=False,
            feedback=feedback,
        )
