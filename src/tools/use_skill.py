"""
On-demand skill loader tool.

The tool lists only skill metadata in the system prompt. Full SKILL.md content
is returned only when the model explicitly calls use_skill.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from src.core.tool import Tool, ToolUseContext
from src.core.types import ToolResult, ValidationResult
from src.skills.loader import Skill


class UseSkillTool(Tool):
    name = "use_skill"
    description = (
        "Load a local SearchClaw skill by name. Skills provide specialized "
        "instructions or domain knowledge for the current task. Call this "
        "before answering when a listed skill matches the user's request."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "skill": {
                "type": "string",
                "description": "The skill name to load.",
            },
            "args": {
                "type": "string",
                "description": "Optional arguments passed to the skill as $ARGUMENTS.",
            },
        },
        "required": ["skill"],
    }

    is_concurrency_safe = False
    is_read_only = True

    def __init__(
        self,
        skills: Iterable[Skill],
        listing_max_chars: int = 8000,
        max_skill_chars: int = 50000,
    ):
        self._skills = {skill.name: skill for skill in skills}
        self.listing_max_chars = listing_max_chars
        self.max_result_size_chars = max_skill_chars

    @property
    def available_skills(self) -> list[Skill]:
        return sorted(
            (skill for skill in self._skills.values() if not skill.disabled),
            key=lambda skill: skill.name,
        )

    def prompt(self) -> str:
        return (
            "Use this tool to load specialized local skills on demand. Skills "
            "provide task-specific instructions and context.\n\n"
            "When a skill matches the user's request, call use_skill before "
            "answering or doing the task. After a skill is loaded, follow its "
            "instructions for the current task and do not call the same skill "
            "again unless the arguments need to change.\n\n"
            "Available skills:\n"
            f"{self._format_skill_listing()}"
        )

    def validate_input(self, args: dict) -> ValidationResult:
        skill_name = self._normalize_skill_name(args.get("skill", ""))
        if not skill_name:
            return ValidationResult(valid=False, message="skill is required")

        skill = self._skills.get(skill_name)
        if skill is None:
            return ValidationResult(
                valid=False,
                message=(
                    f"Unknown skill '{skill_name}'. Available skills: "
                    f"{self._available_skill_names()}"
                ),
            )
        if skill.disabled:
            return ValidationResult(
                valid=False,
                message=f"Skill '{skill_name}' is disabled for model invocation",
            )

        skill_args = args.get("args", "")
        if skill_args is not None and not isinstance(skill_args, str):
            return ValidationResult(valid=False, message="args must be a string")

        return ValidationResult(valid=True)

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        skill_name = self._normalize_skill_name(args.get("skill", ""))
        validation = self.validate_input(args)
        if not validation.valid:
            return ToolResult(data=validation.message, is_error=True)

        skill = self._skills[skill_name]
        skill_args = args.get("args") or ""
        content = self._render_skill_content(skill, skill_args)
        truncated = False
        if len(content) > self.max_result_size_chars:
            content = content[: self.max_result_size_chars]
            content += (
                "\n\n---\n"
                f"[Skill content truncated at {self.max_result_size_chars:,} chars. "
                "Please shorten the skill if the omitted instructions matter.]"
            )
            truncated = True

        return ToolResult(
            data=(
                f"Skill '{skill.name}' loaded. Treat the following content as "
                "instructions for the current task. Do not call this skill "
                "again unless the arguments need to change.\n\n"
                f"{content}"
            ),
            truncated=truncated,
        )

    def _format_skill_listing(self) -> str:
        skills = self.available_skills
        if not skills:
            return "- No local skills are currently available."

        lines: list[str] = []
        used_chars = 0
        for i, skill in enumerate(skills):
            line = self._format_skill_line(skill)
            line_len = len(line) + (1 if lines else 0)
            if used_chars + line_len > self.listing_max_chars:
                remaining = len(skills) - i
                suffix = f"- ... ({remaining} more skill(s) omitted)"
                if used_chars + len(suffix) + 1 <= self.listing_max_chars:
                    lines.append(suffix)
                break
            lines.append(line)
            used_chars += line_len

        return "\n".join(lines)

    def _format_skill_line(self, skill: Skill) -> str:
        description = skill.description.strip()
        if skill.when_to_use:
            description = f"{description} - {skill.when_to_use.strip()}"
        description = " ".join(description.split())
        if len(description) > 250:
            description = description[:247].rstrip() + "..."
        return f"- {skill.name}: {description}"

    def _available_skill_names(self) -> str:
        names = [skill.name for skill in self.available_skills]
        return ", ".join(names) if names else "(none)"

    def _render_skill_content(self, skill: Skill, args: str) -> str:
        skill_dir = _path_for_prompt(skill.base_dir)
        content = skill.content
        content = content.replace("$ARGUMENTS", args)
        content = content.replace("${SKILL_DIR}", skill_dir)
        content = content.replace("${SEARCHCLAW_SKILL_DIR}", skill_dir)
        return f"Base directory for this skill: {skill_dir}\n\n{content}"

    def _normalize_skill_name(self, raw_name: object) -> str:
        if raw_name is None:
            return ""
        return str(raw_name).strip().lstrip("/")


def _path_for_prompt(path: Path) -> str:
    # Match Claude-style skill paths: stable and slash-separated in prompts.
    return str(path).replace("\\", "/")
