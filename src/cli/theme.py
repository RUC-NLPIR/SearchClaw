"""Shared visual theme for the SearchClaw CLI.

Centralizes the color palette, tool icon/color map, and a few rich helpers
so the look stays consistent across render.py and app.py. Changing a color
here changes it everywhere.
"""

from __future__ import annotations

from rich.theme import Theme

# --- Core palette -------------------------------------------------------
# Cyan→blue brand gradient (matches the startup logo), plus semantic colors.
BRAND = "#00afff"
BRAND_DIM = "#0087ff"
ACCENT = "#5fd7ff"
LOGO_GRADIENT = ["#5fd7ff", "#00afff", "#0087ff", "#005fff"]

# Horizontal blue gradient for the solid logo (left→right, light→deep).
# More stops = smoother sweep across the wide wordmark.
LOGO_SWEEP = [
    "#7fdbff", "#5fd7ff", "#33c4ff", "#19b6ff",
    "#00afff", "#0096ff", "#0087ff", "#0072ff", "#005fff",
]
# Dim navy used for the figlet shadow/box-drawing characters, so the solid
# blocks pop forward and the wordmark reads as 3-D.
LOGO_SHADOW = "#1c3a66"

SUCCESS = "#5fd75f"
WARNING = "#ffd75f"
ERROR = "#ff5f5f"
MUTED = "grey50"

# rich Theme — style names usable as [style]...[/style] markup everywhere
# a themed Console is used.
THEME = Theme({
    "brand": f"bold {BRAND}",
    "accent": ACCENT,
    "muted": MUTED,
    "success": SUCCESS,
    "warning": WARNING,
    "error": f"bold {ERROR}",
    "tool": f"bold {ACCENT}",
    "tool.args": "grey58",
    "tool.result": "grey54",
    "prompt": f"bold {BRAND}",
    "source.idx": MUTED,
    "source.title": "white",
    "source.url": BRAND_DIM,
    "stat": "grey62",
    "plan.done": SUCCESS,
    "plan.active": ACCENT,
    "plan.pending": MUTED,
    "dash.border": BRAND_DIM,
    "dash.divider": "grey30",
})

# --- Tool presentation --------------------------------------------------
# icon + display color per tool. Unknown tools fall back to DEFAULT.
TOOL_STYLE: dict[str, tuple[str, str]] = {
    "search_web": ("🔍", ACCENT),
    "academic_search": ("🎓", ACCENT),
    "news_search": ("📰", ACCENT),
    "wechat_search": ("💬", ACCENT),
    "fetch_url": ("📄", BRAND),
    "deep_read": ("🔬", BRAND),
    "local_glob": ("📂", ACCENT),
    "local_search": ("📁", ACCENT),
    "local_read": ("📄", BRAND),
    "cite_source": ("📌", SUCCESS),
    "research_plan": ("📋", WARNING),
    "ask_user": ("❓", WARNING),
    "use_skill": ("🧩", BRAND_DIM),
    "run_skill_script": ("🔧", BRAND_DIM),
}
_DEFAULT_TOOL = ("⏺", ACCENT)

# Plan task status → (icon, theme style)
PLAN_STATUS = {
    "completed": ("●", "plan.done"),
    "in_progress": ("◉", "plan.active"),
    "pending": ("○", "plan.pending"),
}


def tool_style(name: str) -> tuple[str, str]:
    """Return (icon, color) for a tool name."""
    return TOOL_STYLE.get(name, _DEFAULT_TOOL)
