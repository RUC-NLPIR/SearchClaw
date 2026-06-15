"""
Run Python scripts bundled with local skills.

This intentionally does not execute arbitrary shell. The model must name a
loaded skill and a relative .py file inside that skill directory.
"""

from __future__ import annotations

import asyncio
import copy
import sys
from pathlib import Path
from typing import Iterable

from src.core.tool import Tool, ToolUseContext
from src.core.types import ToolResult, ValidationResult
from src.skills.loader import Skill


class RunSkillScriptTool(Tool):
    name = "run_skill_script"
    description = (
        "Run a Python script bundled inside a local SearchClaw skill directory. "
        "Use this only for scripts documented by a skill that has been loaded "
        "with use_skill."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "skill": {
                "type": "string",
                "description": "The skill name whose directory contains the script.",
            },
            "script": {
                "type": "string",
                "description": (
                    "Relative path to a .py script inside the skill directory, "
                    "for example 'scripts/analyze.py'. Absolute paths and '..' "
                    "path traversal are rejected."
                ),
            },
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional command-line arguments passed as argv strings.",
                "default": [],
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Optional timeout in seconds.",
                "default": 30,
                "minimum": 1,
            },
        },
        "required": ["skill", "script"],
    }

    is_concurrency_safe = False
    is_read_only = False

    def __init__(
        self,
        skills: Iterable[Skill],
        default_timeout_seconds: int = 30,
        max_output_chars: int = 20000,
        max_args: int = 50,
        max_arg_chars: int = 1000,
    ):
        self.input_schema = copy.deepcopy(self.input_schema)
        self._skills = {skill.name: skill for skill in skills}
        self.default_timeout_seconds = max(1, default_timeout_seconds)
        self.max_result_size_chars = max_output_chars
        self.max_args = max_args
        self.max_arg_chars = max_arg_chars
        self.input_schema["properties"]["timeout_seconds"]["default"] = self.default_timeout_seconds
        self.input_schema["properties"]["timeout_seconds"]["maximum"] = self.default_timeout_seconds

    def prompt(self) -> str:
        return (
            "Use run_skill_script only after loading a relevant skill with "
            "use_skill and only when that skill's instructions document a "
            "bundled Python script to run.\n\n"
            "Safety rules:\n"
            "- script must be a relative path inside the named skill directory\n"
            "- only .py files are supported\n"
            "- args must be a JSON array of strings; no shell is used\n"
            "- the process runs with the skill directory as its working directory\n"
            "- stdout/stderr are returned and may be truncated"
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

        script = args.get("script", "")
        script_validation = self._validate_script(skill, script)
        if not script_validation.valid:
            return script_validation

        script_args = args.get("args", [])
        args_validation = self._validate_script_args(script_args)
        if not args_validation.valid:
            return args_validation

        timeout = args.get("timeout_seconds", self.default_timeout_seconds)
        if not isinstance(timeout, int):
            return ValidationResult(valid=False, message="timeout_seconds must be an integer")
        if timeout < 1 or timeout > self.default_timeout_seconds:
            return ValidationResult(
                valid=False,
                message=f"timeout_seconds must be between 1 and {self.default_timeout_seconds}",
            )

        return ValidationResult(valid=True)

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        validation = self.validate_input(args)
        if not validation.valid:
            return ToolResult(data=validation.message, is_error=True)

        skill_name = self._normalize_skill_name(args["skill"])
        skill = self._skills[skill_name]
        script_path = self._resolve_script_path(skill, args["script"])
        script_args = list(args.get("args", []))
        timeout = int(args.get("timeout_seconds", self.default_timeout_seconds))

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(script_path),
                *script_args,
                cwd=str(skill.base_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            communicate_task = asyncio.create_task(process.communicate())
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    asyncio.shield(communicate_task),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                stdout_b, stderr_b = await communicate_task
                data, truncated = self._format_result(
                    skill=skill,
                    script_path=script_path,
                    args=script_args,
                    returncode=None,
                    stdout=stdout_b.decode("utf-8", errors="replace"),
                    stderr=stderr_b.decode("utf-8", errors="replace"),
                    timed_out=True,
                    timeout=timeout,
                )
                return ToolResult(data=data, is_error=True, truncated=truncated)

            data, truncated = self._format_result(
                skill=skill,
                script_path=script_path,
                args=script_args,
                returncode=process.returncode,
                stdout=stdout_b.decode("utf-8", errors="replace"),
                stderr=stderr_b.decode("utf-8", errors="replace"),
                timed_out=False,
                timeout=timeout,
            )
            return ToolResult(
                data=data,
                is_error=(process.returncode != 0),
                truncated=truncated,
            )
        except Exception as e:
            return ToolResult(
                data=f"Failed to run skill script: {str(e)}",
                is_error=True,
            )

    def _validate_script(self, skill: Skill, script: object) -> ValidationResult:
        if not isinstance(script, str) or not script.strip():
            return ValidationResult(valid=False, message="script is required")
        try:
            script_path = self._resolve_script_path(skill, script)
        except ValueError as e:
            return ValidationResult(valid=False, message=str(e))

        if script_path.suffix != ".py":
            return ValidationResult(valid=False, message="Only .py skill scripts are supported")
        if not script_path.exists():
            return ValidationResult(valid=False, message=f"Script not found: {script}")
        if not script_path.is_file():
            return ValidationResult(valid=False, message=f"Script is not a file: {script}")
        return ValidationResult(valid=True)

    def _validate_script_args(self, script_args: object) -> ValidationResult:
        if script_args is None:
            return ValidationResult(valid=False, message="args must be an array of strings")
        if not isinstance(script_args, list):
            return ValidationResult(valid=False, message="args must be an array of strings")
        if len(script_args) > self.max_args:
            return ValidationResult(valid=False, message=f"args may contain at most {self.max_args} items")
        for i, arg in enumerate(script_args):
            if not isinstance(arg, str):
                return ValidationResult(valid=False, message=f"args[{i}] must be a string")
            if "\x00" in arg:
                return ValidationResult(valid=False, message=f"args[{i}] contains a NUL byte")
            if len(arg) > self.max_arg_chars:
                return ValidationResult(
                    valid=False,
                    message=f"args[{i}] exceeds {self.max_arg_chars} characters",
                )
        return ValidationResult(valid=True)

    def _resolve_script_path(self, skill: Skill, script: object) -> Path:
        script_raw = str(script).strip()
        script_path = Path(script_raw)
        if script_path.is_absolute():
            raise ValueError("script must be a relative path inside the skill directory")
        if any(part in ("", ".", "..") for part in script_path.parts):
            raise ValueError("script path must not contain empty, '.', or '..' components")

        base_dir = skill.base_dir.resolve()
        resolved = (base_dir / script_path).resolve()
        try:
            resolved.relative_to(base_dir)
        except ValueError as e:
            raise ValueError("script must stay inside the skill directory") from e
        return resolved

    def _format_result(
        self,
        skill: Skill,
        script_path: Path,
        args: list[str],
        returncode: int | None,
        stdout: str,
        stderr: str,
        timed_out: bool,
        timeout: int,
    ) -> tuple[str, bool]:
        relative_script = script_path.relative_to(skill.base_dir.resolve())
        status = "timed out" if timed_out else f"exit code {returncode}"
        output = (
            f"Skill script result ({status})\n"
            f"Skill: {skill.name}\n"
            f"Script: {relative_script}\n"
            f"Args: {args}\n"
            f"Working directory: {skill.base_dir}\n"
            f"Timeout: {timeout}s\n\n"
            "STDOUT:\n"
            f"{stdout.strip() or '(empty)'}\n\n"
            "STDERR:\n"
            f"{stderr.strip() or '(empty)'}"
        )
        if len(output) <= self.max_result_size_chars:
            return output, False
        truncated = output[: self.max_result_size_chars]
        truncated += (
            "\n\n---\n"
            f"[Script output truncated at {self.max_result_size_chars:,} chars.]"
        )
        return truncated, True

    def _available_skill_names(self) -> str:
        names = sorted(skill.name for skill in self._skills.values() if not skill.disabled)
        return ", ".join(names) if names else "(none)"

    def _normalize_skill_name(self, raw_name: object) -> str:
        if raw_name is None:
            return ""
        return str(raw_name).strip().lstrip("/")
