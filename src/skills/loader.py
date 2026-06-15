"""
Local SKILL.md loader.

Skills are discovered from directories shaped like:

    skills/<skill-name>/SKILL.md

Only metadata is intended for prompt listings; the full markdown body is
loaded on demand by the use_skill tool.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Skill:
    """A local skill loaded from a SKILL.md file."""

    name: str
    description: str
    when_to_use: str
    content: str
    base_dir: Path
    path: Path
    disabled: bool = False


def load_skills(
    dirs: Iterable[str | Path],
    root: Path | None = None,
) -> list[Skill]:
    """
    Load skills from one or more skill roots.

    Later directories override earlier ones when two skills resolve to the
    same name. Missing directories are ignored.
    """
    base_root = (root or Path.cwd()).resolve()
    skills_by_name: dict[str, Skill] = {}

    for raw_dir in dirs:
        skill_root = _resolve_dir(raw_dir, base_root)
        for skill in _load_skills_from_dir(skill_root):
            if skill.name in skills_by_name:
                logger.info(
                    "Skill '%s' from %s overrides earlier definition from %s",
                    skill.name,
                    skill.path,
                    skills_by_name[skill.name].path,
                )
            skills_by_name[skill.name] = skill

    return list(skills_by_name.values())


def _resolve_dir(raw_dir: str | Path, root: Path) -> Path:
    path = Path(raw_dir).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _load_skills_from_dir(skill_root: Path) -> list[Skill]:
    if not skill_root.exists():
        logger.debug("Skill directory not found: %s", skill_root)
        return []
    if not skill_root.is_dir():
        logger.warning("Configured skill path is not a directory: %s", skill_root)
        return []

    skills: list[Skill] = []
    try:
        entries = sorted(skill_root.iterdir(), key=lambda p: p.name)
    except OSError as e:
        logger.warning("Failed to list skill directory %s: %s", skill_root, e)
        return []

    for entry in entries:
        if not entry.is_dir():
            continue
        skill_path = entry / "SKILL.md"
        if not skill_path.exists():
            continue
        try:
            skills.append(_load_skill(entry, skill_path))
        except Exception as e:
            logger.warning("Failed to load skill %s: %s", skill_path, e)

    return skills


def _load_skill(base_dir: Path, skill_path: Path) -> Skill:
    raw = skill_path.read_text(encoding="utf-8")
    frontmatter, content = _split_frontmatter(raw)

    dir_name = base_dir.name.strip()
    configured_name = _string_field(frontmatter.get("name"))
    name = _normalize_name(configured_name or dir_name)
    if not name:
        raise ValueError(f"Skill name is empty for {skill_path}")

    description = _string_field(frontmatter.get("description"))
    if not description:
        description = _fallback_description(content, name)

    when_to_use = _string_field(frontmatter.get("when_to_use"))
    disabled = _bool_field(frontmatter.get("disable-model-invocation"))

    return Skill(
        name=name,
        description=description,
        when_to_use=when_to_use,
        content=content.strip(),
        base_dir=base_dir.resolve(),
        path=skill_path.resolve(),
        disabled=disabled,
    )


def _split_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, raw

    closing_index = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            closing_index = i
            break

    if closing_index is None:
        return {}, raw

    frontmatter_text = "\n".join(lines[1:closing_index])
    body = "\n".join(lines[closing_index + 1 :])

    try:
        parsed = yaml.safe_load(frontmatter_text) or {}
    except yaml.YAMLError as e:
        logger.warning("Invalid skill frontmatter ignored: %s", e)
        return {}, body

    if not isinstance(parsed, dict):
        logger.warning("Skill frontmatter must be a mapping; ignoring %r", type(parsed).__name__)
        return {}, body

    return parsed, body


def _string_field(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _bool_field(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _normalize_name(name: str) -> str:
    # Tool inputs are string names, so keep the format permissive but strip
    # whitespace and a leading slash for slash-command compatibility.
    return name.strip().lstrip("/")


def _fallback_description(content: str, name: str) -> str:
    for line in content.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        cleaned = re.sub(r"^#{1,6}\s*", "", cleaned).strip()
        cleaned = cleaned.strip("*_` ")
        if cleaned:
            return cleaned[:250]
    return f"Skill instructions for {name}"
