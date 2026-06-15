"""Custom Textual widgets for the SearchClaw TUI.

Three widgets compose the layout:
  - PlanPanel: top, shows the research plan (x/total + progress bar + tasks)
  - ActivityPanel: above the input, shows the current turn's live progress
    (tool calls, status, spinner) in a fixed-height region
  - ChatLog: the scrollable middle region where each turn's question and
    final report stack chat-style

Colors/icons reuse the constants in src/cli/theme.py so the look matches the
rest of the CLI.
"""

from __future__ import annotations

from collections import deque

from rich.text import Text
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Static

from src.cli.theme import ACCENT, BRAND, LOGO_SHADOW, LOGO_SWEEP, MUTED, SUCCESS

# Solid figlet wordmark (ansi_shadow font). Box-drawing glyphs form the
# drop-shadow; they're dimmed while a left→right blue gradient sweeps the solid
# blocks, so the wordmark reads as 3-D.
_LOGO = [
    "███████╗███████╗ █████╗ ██████╗  ██████╗██╗  ██╗ ██████╗██╗      █████╗ ██╗    ██╗",
    "██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝██║  ██║██╔════╝██║     ██╔══██╗██║    ██║",
    "███████╗█████╗  ███████║██████╔╝██║     ███████║██║     ██║     ███████║██║ █╗ ██║",
    "╚════██║██╔══╝  ██╔══██║██╔══██╗██║     ██╔══██║██║     ██║     ██╔══██║██║███╗██║",
    "███████║███████╗██║  ██║██║  ██║╚██████╗██║  ██║╚██████╗███████╗██║  ██║╚███╔███╔╝",
    "╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝ ╚═════╝╚══════╝╚═╝  ╚═╝ ╚══╝╚══╝ ",
]
_LOGO_WIDTH = max(len(row) for row in _LOGO)
_LOGO_SMALL = "SearchClaw"
_SHADOW_CHARS = set("╗╚╝═║╔╣╠╦╩╬")


def _gradient_logo() -> Text:
    """The wide wordmark with a horizontal blue sweep + dim shadow."""
    out = Text(justify="left")
    n = len(LOGO_SWEEP)
    for row in _LOGO:
        for col, ch in enumerate(row):
            if ch == " ":
                out.append(" ")
            elif ch in _SHADOW_CHARS:
                out.append(ch, style=LOGO_SHADOW)
            else:
                color = LOGO_SWEEP[min(n - 1, col * n // _LOGO_WIDTH)]
                out.append(ch, style=f"bold {color}")
        out.append("\n")
    return out


def _small_logo() -> Text:
    out = Text()
    n = len(LOGO_SWEEP)
    for i, ch in enumerate(_LOGO_SMALL):
        color = LOGO_SWEEP[min(n - 1, i * n // len(_LOGO_SMALL))]
        out.append(ch, style=f"bold {color}")
    return out


class WelcomeBanner(Static):
    """Startup banner: gradient logo + tagline + model line.

    Rendered via render() (called after layout, so the widget's width is
    known) to pick the wide art or a compact wordmark. Removed by the app the
    first time the user submits input.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._model = ""

    def set_model(self, model: str) -> None:
        self._model = model
        self.refresh(layout=True)

    def render(self) -> Text:
        wide = self.size.width >= _LOGO_WIDTH + 2 or self.size.width == 0
        out = Text()
        out.append_text(_gradient_logo() if wide else _small_logo())
        out.append("\n")
        out.append("◆ web research agent\n", style="dim italic")
        out.append(f"model {self._model}   ·   /help for commands   ·   Ctrl+D to quit", style="grey50")
        return out


# How many recent activity lines stay visible in the bottom panel.
ACTIVITY_WINDOW = 6

# Plan status → (icon, concrete color). theme.PLAN_STATUS uses rich *theme
# style names* (e.g. "plan.active") which Textual can't resolve, so the TUI
# maps to real colors here.
_PLAN_STATUS = {
    "completed": ("●", SUCCESS),
    "in_progress": ("◉", ACCENT),
    "pending": ("○", MUTED),
}


def _progress_bar(done: int, total: int, width: int = 16) -> Text:
    """A unicode progress bar as a rich Text."""
    if total <= 0:
        return Text("")
    filled = round(width * done / total)
    bar = Text()
    bar.append("█" * filled, style=SUCCESS)
    bar.append("░" * (width - filled), style=MUTED)
    return bar


class PlanPanel(Static):
    """Top panel: research plan overview. Hidden until a plan arrives."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._tasks: list[dict] = []
        self._done = 0
        self._total = 0

    def update_plan(self, tasks: list[dict], done: int, total: int) -> None:
        self._tasks = tasks or []
        self._done = done
        self._total = total
        self.display = total > 0
        self.refresh(layout=True)

    def clear_plan(self) -> None:
        self._tasks, self._done, self._total = [], 0, 0
        self.display = False

    def collapse(self) -> None:
        """Hide the panel but keep its data.

        Called when a turn finishes so the report has the full screen. The
        next turn's PLAN_UPDATE re-shows it; /clear wipes it via clear_plan.
        """
        self.display = False

    def render(self) -> Text:
        out = Text()
        out.append("Research Plan ", style=f"bold {BRAND}")
        out.append(f"{self._done}/{self._total}  ", style="grey62")
        out.append_text(_progress_bar(self._done, self._total))
        out.append("\n")
        for t in self._tasks:
            icon, style = _PLAN_STATUS.get(t.get("status", "pending"), _PLAN_STATUS["pending"])
            title = t.get("title", "")
            title_style = "white" if t.get("status") != "pending" else MUTED
            out.append(f"  {icon} ", style=style)
            out.append(title + "\n", style=title_style)
        return out


class ActivityPanel(Static):
    """Bottom panel (above input): live progress for the current turn.

    Holds a fixed-height rolling window of recent tool/status lines plus a
    final status line. Cleared between turns.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._lines: deque[Text] = deque(maxlen=ACTIVITY_WINDOW)
        self._status = ""

    def log_line(self, line: Text) -> None:
        self._lines.append(line)
        self.refresh(layout=True)

    def set_status(self, text: str) -> None:
        self._status = text
        self.refresh(layout=True)

    def clear(self) -> None:
        self._lines.clear()
        self._status = ""
        self.display = False

    def begin(self) -> None:
        """Show the panel at the start of a turn."""
        self._lines.clear()
        self._status = "working…"
        self.display = True
        self.refresh(layout=True)

    def render(self) -> Text:
        out = Text()
        for line in self._lines:
            out.append_text(line)
            out.append("\n")
        if self._status:
            out.append("⠿ ", style=ACCENT)
            out.append(self._status, style=MUTED)
        return out


class ChatLog(VerticalScroll):
    """Scrollable chat-style log of questions + reports.

    Unlike RichLog (which flattens content into static strips and so can't
    carry clickable links), this mounts each entry as its own child widget.
    That lets reports be real Textual ``Markdown`` widgets whose links open in
    the browser. Plain lines (Text/str) are mounted as ``Static``.
    """

    def write(self, renderable, scroll: bool = True) -> None:
        """Mount a renderable as a new child.

        - a Textual ``Widget`` (e.g. a ``Markdown``) is mounted as-is
        - a rich ``Text`` / ``str`` becomes a ``Static`` line
        """
        if isinstance(renderable, Widget):
            child = renderable
        else:
            content = renderable if renderable != "" else Text("")
            child = Static(content, classes="chat-line")
        self.mount(child)
        if scroll:
            self.scroll_end(animate=False)

    def clear(self) -> None:
        """Remove all entries."""
        for child in list(self.children):
            child.remove()


# (StreamView shows the full streaming answer and auto-scrolls; no tail cap.)


class StreamView(Static):
    """Live, plain-text view of the answer as it streams in.

    RichLog can't rewrite already-written content, and half-finished Markdown
    (an unclosed code fence, a half-drawn table) renders broken. So while the
    answer streams we show the raw text here, filling the middle region and
    auto-scrolling to follow the newest text. On completion the app clears this
    and writes the final, fully rendered Markdown into the ChatLog. Hidden when
    there's nothing streaming.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._text = ""

    def begin(self) -> None:
        self._text = ""
        self.display = False

    def append(self, chunk: str) -> None:
        self._text += chunk
        self.display = bool(self._text)
        self.update(self._render_text())
        # Follow the tail as new text arrives.
        self.scroll_end(animate=False)

    def clear(self) -> None:
        self._text = ""
        self.display = False
        self.update(Text(""))

    def _render_text(self) -> Text:
        if not self._text:
            return Text("")
        return Text(self._text, style="white")

