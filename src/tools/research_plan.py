"""
Research plan tool — structured task decomposition for complex queries.

Inspired by Claude Code's TodoWrite system. Lets the LLM decompose
complex, multi-part research questions into tracked sub-tasks, work
through them systematically, and record findings as it goes.

A single tool with action={create, update, check} to keep the tool
count small. State lives on LoopState.research_plan (in-memory, single
session — no file persistence needed).

The companion PlanCompletenessHook (stop hook) ensures the agent doesn't
finalize until all sub-tasks are completed.
"""

from __future__ import annotations

import logging

from src.core.tool import Tool, ToolUseContext
from src.core.types import (
    ResearchPlan,
    ResearchTask,
    ToolResult,
    ValidationResult,
)

logger = logging.getLogger(__name__)


class ResearchPlanTool(Tool):
    name = "research_plan"
    description = (
        "Create and manage a structured research plan for complex queries. "
        "Use action='create' to decompose a question into sub-tasks, "
        "action='update' to record progress and findings, "
        "action='check' to view current plan status."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "update", "check"],
                "description": (
                    "create: decompose query into sub-tasks. "
                    "update: mark a sub-task's progress and record findings. "
                    "check: view current plan status."
                ),
            },
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Short title for the sub-task.",
                        },
                        "details": {
                            "type": "string",
                            "description": "Additional context or search strategy for this sub-task.",
                        },
                    },
                    "required": ["title"],
                },
                "description": "Sub-tasks to create (only for action='create').",
            },
            "task_id": {
                "type": "string",
                "description": "Task ID to update (only for action='update').",
            },
            "status": {
                "type": "string",
                "enum": ["in_progress", "completed"],
                "description": "New status for the task (only for action='update').",
            },
            "findings": {
                "type": "string",
                "description": (
                    "Key findings for this sub-task. Record what you learned — "
                    "these help you synthesize the final answer later (for action='update')."
                ),
            },
        },
        "required": ["action"],
    }

    is_concurrency_safe = False   # Mutates LoopState — must be sequential
    is_read_only = False
    max_result_size_chars = 10000

    def prompt(self) -> str:
        return """Use this tool to create a structured research plan before starting your search. Breaking complex questions into sub-tasks ensures comprehensive coverage and prevents forgetting sub-topics.

## When to Use
- Any question with 2+ aspects, topics, or dimensions to investigate
- Comparative questions ("compare X and Y", "differences between")
- Analytical questions ("how does X work", "why did X happen")
- Questions requiring breadth ("what are the main...", "overview of...")
- Multi-step research where later steps depend on earlier findings

## When NOT to Use
- Simple single-fact lookups ("Who is X?", "What is the capital of Y?")
- Questions that can be fully answered with a single search

## Workflow
1. FIRST: Call with action="create" to decompose the query into 3-7 sub-tasks
2. Work through each sub-task: search, fetch, read sources
3. Call with action="update" to record findings after completing each sub-task
4. After ALL sub-tasks are completed, synthesize the final answer

## Examples

<example>
User: "Compare the economic policies of US, EU, and China on AI regulation"
<reasoning>
This question involves 3 distinct regions and multiple policy dimensions.
Without a plan, the agent would likely research the US thoroughly but
give shallow treatment to EU and China. Creating a plan ensures balanced coverage.
</reasoning>
Action: Create plan with sub-tasks for each region + a comparison synthesis task.
</example>

<example>
User: "What are the causes, effects, and proposed solutions for ocean plastic pollution?"
<reasoning>
Three distinct research angles (causes, effects, solutions) each requiring
different searches and sources. A plan prevents the agent from going deep
on causes and forgetting to research solutions.
</reasoning>
Action: Create plan with sub-tasks for causes, effects, solutions, and synthesis.
</example>

<example>
User: "Who won the 2024 US presidential election?"
<reasoning>
Simple factual lookup. One search will answer this. No plan needed.
</reasoning>
Action: Skip plan, search directly.
</example>

<example>
User: "What is the latest version of Python?"
<reasoning>
Single-fact lookup with one clear search target. No plan needed.
</reasoning>
Action: Skip plan, search directly.
</example>

## Tips
- Create 3-7 sub-tasks (each should map to 1-2 searches)
- Record key findings when marking a task completed — these help you synthesize later
- Call action="check" at any time to review your progress"""

    def validate_input(self, args: dict) -> ValidationResult:
        action = args.get("action")
        if action not in ("create", "update", "check"):
            return ValidationResult(
                valid=False,
                message="action must be one of: create, update, check",
            )
        if action == "create":
            tasks = args.get("tasks")
            if not tasks or not isinstance(tasks, list) or len(tasks) == 0:
                return ValidationResult(
                    valid=False,
                    message="action='create' requires a non-empty 'tasks' array",
                )
            for i, t in enumerate(tasks):
                if not t.get("title"):
                    return ValidationResult(
                        valid=False,
                        message=f"Task {i} is missing a 'title'",
                    )
        if action == "update":
            if not args.get("task_id"):
                return ValidationResult(
                    valid=False,
                    message="action='update' requires 'task_id'",
                )
            if not args.get("status"):
                return ValidationResult(
                    valid=False,
                    message="action='update' requires 'status' (in_progress or completed)",
                )
        return ValidationResult(valid=True)

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        action = args["action"]

        # Get LoopState reference from context.extra
        loop_state = context.extra.get("loop_state")
        if loop_state is None:
            return ToolResult(
                data="Error: research_plan tool requires loop_state in context. This is a bug.",
                is_error=True,
            )

        if action == "create":
            return self._create_plan(args, loop_state)
        elif action == "update":
            return self._update_plan(args, loop_state)
        elif action == "check":
            return self._check_plan(loop_state)
        else:
            return ToolResult(data=f"Unknown action: {action}", is_error=True)

    def _create_plan(self, args: dict, loop_state) -> ToolResult:
        """Create a new research plan from a list of sub-tasks."""
        tasks_input = args["tasks"]

        # Build ResearchTask objects with auto-assigned IDs
        tasks = []
        for i, t in enumerate(tasks_input, start=1):
            tasks.append(ResearchTask(
                id=str(i),
                title=t["title"],
                details=t.get("details", ""),
                status="pending",
            ))

        plan = ResearchPlan(tasks=tasks)
        loop_state.research_plan = plan

        logger.info(f"Research plan created with {len(tasks)} sub-tasks")

        summary = f"Research plan created with {len(tasks)} sub-tasks:\n\n"
        summary += plan.summary()
        summary += "\n\nWork through each sub-task by searching and reading sources, "
        summary += "then call research_plan(action='update') to record your findings."

        return ToolResult(data=summary)

    def _update_plan(self, args: dict, loop_state) -> ToolResult:
        """Update a sub-task's status and record findings."""
        plan = loop_state.research_plan
        if plan is None:
            return ToolResult(
                data="No research plan exists. Call with action='create' first.",
                is_error=True,
            )

        task_id = args["task_id"]
        task = plan.get_task(task_id)
        if task is None:
            available = ", ".join(t.id for t in plan.tasks)
            return ToolResult(
                data=f"Task '{task_id}' not found. Available IDs: {available}",
                is_error=True,
            )

        new_status = args["status"]
        findings = args.get("findings", "")

        task.status = new_status
        if findings:
            task.findings = findings

        logger.info(f"Task [{task_id}] '{task.title}' → {new_status}")

        summary = f"Task [{task_id}] updated to '{new_status}'.\n\n"
        summary += plan.summary()

        if plan.is_complete:
            summary += "\n\nAll sub-tasks completed! You can now synthesize your final answer."

        return ToolResult(data=summary)

    def _check_plan(self, loop_state) -> ToolResult:
        """Return current plan status."""
        plan = loop_state.research_plan
        if plan is None:
            return ToolResult(data="No research plan exists yet.")

        summary = "Current research plan:\n\n"
        summary += plan.summary()

        if plan.is_complete:
            summary += "\n\nAll sub-tasks completed! Ready to synthesize final answer."
        else:
            pending = [t for t in plan.tasks if t.status != "completed"]
            summary += f"\n\n{len(pending)} sub-task(s) remaining."

        return ToolResult(data=summary)
