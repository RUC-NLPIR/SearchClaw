"""SearchClaw Textual TUI application.

Full-screen alt-screen UI: a top plan panel, a chat-style scrolling log of
questions + reports, a bottom activity panel for live progress, and a bordered
input box. It drives the same query_loop the rest of the CLI uses; only the
presentation differs from the old inline renderer.

Event flow: on submit, a Textual worker runs query_loop and renders each
StreamEvent into the widgets. USER_QUESTION pauses the worker on an
asyncio.Future that the input box resolves.
"""

from __future__ import annotations

import asyncio
import json
import os

# Must precede any textual import (textual.constants reads it at import time).
# Disables the Kitty keyboard protocol so CJK IME input isn't delivered as raw
# `CSI ...u` escape sequences. See src/cli/app.py for the full explanation.
os.environ.setdefault("TEXTUAL_DISABLE_KITTY_KEY", "1")

from pathlib import Path
from typing import Any

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.suggester import Suggester
from textual.widgets import Input, Markdown

from src.cli import config as cli_config
from src.cli.mentions import parse_mentions
from src.cli.query import CLISession, build_query_params, finalize_turn
from src.cli.runtime import Runtime, build_runtime
from src.cli.theme import ACCENT, SUCCESS, tool_style
from src.cli.tui.widgets import ActivityPanel, ChatLog, PlanPanel, WelcomeBanner
from src.core.loop import query_loop
from src.core.types import EventType

EFFORT_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh", "max")

# Slash commands offered as inline suggestions (command, one-line help).
SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/help", "show commands"),
    ("/clear", "start a new conversation"),
    ("/config", "re-run the setup wizard"),
    ("/model", "show or set the model"),
    ("/effort", "show or set reasoning effort"),
    ("/copy", "copy the last answer to clipboard"),
    ("/export", "export the last answer: /export <path.docx>"),
    ("/roots", "show local-search dirs"),
    ("/sessions", "list recent sessions"),
    ("/load", "resume a session: /load <n>"),
    ("/verbose", "toggle reasoning output"),
    ("/exit", "quit"),
]


class SlashSuggester(Suggester):
    """Inline grey autosuggestion for slash commands.

    Only fires when the input starts with '/', so normal questions are never
    suggested against. Returns the first command that the typed text is a
    prefix of; the Input renders the remainder in grey (accept with →).
    """

    def __init__(self) -> None:
        super().__init__(use_cache=True, case_sensitive=False)

    async def get_suggestion(self, value: str) -> str | None:
        if not value.startswith("/"):
            return None
        low = value.lower()
        for cmd, _help in SLASH_COMMANDS:
            if cmd.startswith(low) and cmd != low:
                return cmd
        return None


class HistoryInput(Input):
    """Input with shell-style Up/Down recall of previously submitted text.

    History is per-session and in-memory. Up walks toward older entries,
    Down toward newer; stepping past the newest restores the draft the user
    was typing before they started browsing.
    """

    BINDINGS = [
        Binding("up", "history_prev", show=False),
        Binding("down", "history_next", show=False),
    ]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._history: list[str] = []
        # Index into _history while browsing; len(_history) means "at the
        # live draft" (not browsing). _draft holds that unsubmitted text.
        self._hist_idx = 0
        self._draft = ""

    def add_history(self, text: str) -> None:
        text = text.strip()
        if text and (not self._history or self._history[-1] != text):
            self._history.append(text)
        # Any submission resets browsing to the live (empty) draft.
        self._hist_idx = len(self._history)
        self._draft = ""

    def action_history_prev(self) -> None:
        if not self._history:
            return
        # Entering history from the live draft: stash what's typed now.
        if self._hist_idx == len(self._history):
            self._draft = self.value
        if self._hist_idx > 0:
            self._hist_idx -= 1
            self._set_value(self._history[self._hist_idx])

    def action_history_next(self) -> None:
        if self._hist_idx >= len(self._history):
            return
        self._hist_idx += 1
        if self._hist_idx == len(self._history):
            self._set_value(self._draft)
        else:
            self._set_value(self._history[self._hist_idx])

    def _set_value(self, value: str) -> None:
        self.value = value
        self.cursor_position = len(value)


def _summarize_args(tool_input: dict[str, Any]) -> str:
    if not tool_input:
        return ""
    parts = []
    for k, v in tool_input.items():
        s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        if len(s) > 60:
            s = s[:57] + "..."
        parts.append(f"{k}={s}")
    return ", ".join(parts)


class SearchClawApp(App):
    """The SearchClaw research TUI."""

    CSS_PATH = "styles.tcss"
    TITLE = "SearchClaw"

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("ctrl+d", "quit", "Quit", show=False),
        Binding("ctrl+y", "copy_selection", "Copy selection", show=False),
    ]

    def __init__(self, runtime: Runtime, verbose: bool = False):
        super().__init__()
        self.sess = CLISession(runtime)
        self.verbose = verbose
        # Set while a turn is mid-flight; resolved by the input box when the
        # agent asks the user a question.
        self._answer_future: asyncio.Future[str] | None = None
        self._busy = False
        # Per-turn accumulators.
        self._answer_buf: list[str] = []
        # The Static widget that streams the current text block live, plus the
        # text accumulated into it. Reset to None between blocks.
        self._live_widget = None
        self._live_buf: list[str] = []
        # The startup logo lives in its own widget and is removed the first
        # time the user submits anything.
        self._welcome_shown = False
        # Sessions listed by the last /sessions, so /load <n> can resolve them.
        self._listed_sessions: list[dict] = []

    # --- layout ------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield PlanPanel(id="plan-panel")
        yield WelcomeBanner(id="welcome")
        yield ChatLog(id="chat-log")
        yield ActivityPanel(id="activity-panel")
        yield HistoryInput(
            placeholder="Ask a research question…  (@path for local files, /help)",
            id="prompt-input",
            suggester=SlashSuggester(),
        )

    def on_mount(self) -> None:
        self.query_one("#plan-panel", PlanPanel).display = False
        self.query_one("#activity-panel", ActivityPanel).display = False
        cfg = self.sess.runtime.llm_client.config
        self.query_one("#welcome", WelcomeBanner).set_model(
            cfg.default_model, cfg.reasoning_effort or "off"
        )
        self._welcome_shown = True
        self.query_one("#prompt-input", Input).focus()

    def _dismiss_welcome(self) -> None:
        """Remove the startup banner once the user starts using the app."""
        if self._welcome_shown:
            self._welcome_shown = False
            try:
                self.query_one("#welcome", WelcomeBanner).remove()
            except Exception:
                pass

    # --- input -------------------------------------------------------

    @on(Input.Submitted, "#prompt-input")
    def _on_submit(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.clear()
        if not text:
            return

        # Record every non-empty submission for Up/Down recall.
        if isinstance(event.input, HistoryInput):
            event.input.add_history(text)

        # If the agent is waiting on a question, this input answers it.
        if self._answer_future is not None and not self._answer_future.done():
            self._answer_future.set_result(text)
            return

        self._submit_text(text)

    def _submit_text(self, text: str) -> None:
        # First real input dismisses the startup banner.
        self._dismiss_welcome()

        if self._busy:
            self._log(Text("Still working on the previous query…", style="grey50"))
            return

        if text.startswith("/"):
            self._handle_slash(text)
            return

        # Pull @path mentions: grant local-search roots, strip from the query.
        cleaned, new_roots, new_focus, warnings = parse_mentions(text)

        # An unresolved @path is almost always a typo or a directory the user
        # expected to exist. Hard-stop the turn: report which path is missing
        # and make the user fix the input. We do NOT fall back to a web-only
        # search, because dropping the local root silently changes the answer.
        if warnings:
            for w in warnings:
                self._log(Text(f"! {w}", style="yellow"))
            self._log(Text(
                "Path not found — nothing was searched. Re-enter your question "
                "with a correct @path (or remove the @ to search the web).",
                style="bold yellow",
            ))
            return

        if new_roots:
            added = [r for r in new_roots if r not in self.sess.allowed_roots]
            self.sess.allowed_roots.extend(added)
            for f in new_focus:
                if f not in self.sess.focus_files:
                    self.sess.focus_files.append(f)
            if added:
                self._log(Text("Local search enabled for: " + ", ".join(added), style="grey50"))
        query = cleaned if new_roots else text

        if not query:
            return

        # Echo the user's *original* input (with the @path) so what they typed
        # stays visible, but send the cleaned query (without the @) to the model.
        self._log_user(text)
        self.run_research(query)

    # --- the research worker ----------------------------------------

    @work(exclusive=True)
    async def run_research(self, query: str) -> None:
        self._busy = True
        self._answer_buf.clear()
        self._live_widget = None
        self._live_buf = []
        activity = self.query_one("#activity-panel", ActivityPanel)
        activity.begin()

        params = await build_query_params(self.sess, query)
        session_summary = None
        done_data = None
        final_messages = None

        gen = query_loop(params)
        sent_value: str | None = None
        delta_since_yield = 0
        try:
            while True:
                event = await gen.asend(sent_value)
                sent_value = None
                if event.type == EventType.USER_QUESTION:
                    sent_value = await self._ask_user(event.data)
                    continue
                self._render_event(event)
                # Let Textual repaint. Without yielding, a dense stream of
                # events keeps the worker busy and panels never repaint until
                # the turn ends. Yield frequently during text streaming so the
                # live answer block paints near character-by-character; yield on
                # every non-text event too.
                if event.type == EventType.TEXT_DELTA:
                    delta_since_yield += 1
                    if delta_since_yield >= 2:
                        delta_since_yield = 0
                        await asyncio.sleep(0)
                else:
                    delta_since_yield = 0
                    await asyncio.sleep(0)
                if event.type == EventType.DONE:
                    done_data = event.data
                    session_summary = event.data.get("session_summary")
                    final_messages = event.data.get("final_messages")
        except StopAsyncIteration:
            pass
        except Exception as e:  # surface worker errors instead of silent death
            self._log(Text(f"Error: {e}", style="bold red"))
        finally:
            activity.clear()
            # Collapse the plan panel so the finished report gets the full
            # screen; its data is kept and the next turn re-shows it.
            self.query_one("#plan-panel", PlanPanel).collapse()
            self._busy = False

        finalize_turn(self.sess, query, session_summary, done_data, final_messages)

    # --- event rendering --------------------------------------------

    def _render_event(self, event) -> None:
        et = event.type
        data = event.data or {}
        activity = self.query_one("#activity-panel", ActivityPanel)

        if et == EventType.REASONING_BLOCKS:
            return

        if et == EventType.TEXT_DELTA:
            text = data.get("text", "")
            if text:
                self._answer_buf.append(text)
                self._stream_delta(text)
                activity.set_status("composing answer…")

        elif et == EventType.REASONING_DELTA:
            activity.set_status("thinking…")

        elif et == EventType.TOOL_USE:
            # Text emitted before a tool call is the model's between-step
            # reasoning ("let me confirm…"), not the answer. Drop the live
            # block so it doesn't linger above the tool activity or bleed into
            # the final report.
            self._discard_live()
            name = data.get("tool_name", "?")
            icon, color = tool_style(name)
            args = _summarize_args(data.get("tool_input", {}))
            line = Text()
            line.append(f"{icon} ", style=color)
            line.append(name, style=f"bold {color}")
            if args:
                line.append(f"  {args}", style="grey58")
            activity.log_line(line)
            activity.set_status(f"running {name}…")

        elif et == EventType.TOOL_RESULT:
            name = data.get("tool_name", "?")
            chars = data.get("result_chars", 0)
            is_error = data.get("is_error", False)
            preview = (data.get("result", "") or "").strip().replace("\n", " ")
            if len(preview) > 90:
                preview = preview[:87] + "..."
            line = Text("  ")
            if is_error:
                line.append("⎿ ", style="red")
                line.append(f"{name} error: ", style="red")
                line.append(preview, style="grey54")
            else:
                line.append("⎿ ", style="grey50")
                line.append(preview, style="grey54")
                if chars:
                    line.append(f"  ({chars:,} chars)", style="grey50")
            activity.log_line(line)

        elif et == EventType.PLAN_UPDATE:
            self.query_one("#plan-panel", PlanPanel).update_plan(
                data.get("tasks", []),
                data.get("completed_count", 0),
                data.get("total_count", 0),
            )

        elif et == EventType.STATUS:
            msg = data.get("message", "")
            if msg:
                activity.set_status(msg)

        elif et == EventType.ERROR:
            self._discard_live()
            self._flush_answer()
            self._log(Text(f"Error: {data.get('message', '')}", style="bold red"))

        elif et == EventType.DONE:
            self._render_done(data)

    def _stream_delta(self, text: str) -> None:
        """Append a text delta to the live streaming block, creating it lazily."""
        chat = self.query_one("#chat-log", ChatLog)
        if self._live_widget is None:
            self._live_widget = chat.mount_live()
            self._live_buf = []
        self._live_buf.append(text)
        self._live_widget.update(Text("".join(self._live_buf)))
        chat.scroll_end(animate=False)

    def _discard_live(self) -> None:
        """Remove the live streaming block (intermediate, pre-tool text)."""
        if self._live_widget is not None:
            try:
                self._live_widget.remove()
            except Exception:
                pass
            self._live_widget = None
            self._live_buf = []

    def _flush_answer(self) -> None:
        text = "".join(self._answer_buf).strip()
        self._answer_buf.clear()
        if text:
            self._log_markdown(text)

    def _render_done(self, data: dict) -> None:
        # Drop the live streaming block (raw text). We re-render the answer as
        # proper Markdown below from the authoritative final_answer in the DONE
        # event — it holds ONLY the last turn's assistant text (loop.py:
        # state.last_assistant_message). The streaming buffer accumulates every
        # turn, including between-tool remarks, so we never render it as final.
        self._discard_live()
        self._answer_buf.clear()
        final_answer = (data.get("session_summary") or {}).get("final_answer", "")
        if final_answer.strip():
            self._log_markdown(final_answer.strip())

        # The model's answer already ends with its own "Sources" section
        # (rendered above as clickable Markdown), matching the web UI. So we
        # only show a compact stats line here — rendering the citation list
        # again would duplicate the sources on screen.
        citations = data.get("citations", [])
        turns = data.get("turn_count", 0)
        n = len(citations)
        self._log(Text(f"{turns} turns · {n} source{'s' if n != 1 else ''}", style="grey62"))
        self._log("")

    # --- ask_user ----------------------------------------------------

    async def _ask_user(self, data: dict) -> str:
        question = data.get("question", "")
        options = data.get("options", [])
        self._log(Text(f"? {question}", style="bold yellow"))
        for i, opt in enumerate(options, 1):
            label = opt.get("label", "")
            desc = opt.get("description", "")
            line = Text(f"  {i}. ", style="bold")
            line.append(label)
            if desc:
                line.append(f" — {desc}", style="grey50")
            self._log(line)
        self._log(Text("  Type a number, or your own answer.", style="grey50"))
        self.query_one("#activity-panel", ActivityPanel).set_status("waiting for your answer…")

        self._answer_future = asyncio.get_running_loop().create_future()
        raw = (await self._answer_future).strip()
        self._answer_future = None

        if not raw:
            return options[0]["label"] if options else ""
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]["label"]
        return raw

    # --- slash commands ---------------------------------------------

    def _handle_slash(self, cmd: str) -> None:
        parts = cmd.strip().split(maxsplit=1)
        name = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if name in ("/exit", "/quit"):
            self.exit()
        elif name == "/help":
            self._log(Text(
                "Commands\n"
                "  /clear     start a new conversation\n"
                "  /config    re-run the setup wizard (endpoint, models, keys)\n"
                "  /model     show or set the model\n"
                "  /effort    show or set reasoning effort: off|low|medium|high|xhigh|max\n"
                "  /copy      copy the last answer (raw markdown) to clipboard\n"
                "  /export    export the last answer to DOCX: /export <path.docx>\n"
                "  /roots     show local-search dirs granted via @path\n"
                "  /sessions  list recent saved sessions\n"
                "  /load      resume a past session: /load <n> (after /sessions)\n"
                "  /verbose   toggle reasoning output\n"
                "  /exit      quit\n"
                "@path grants local search for a dir/file (e.g. @~/docs what does X say)\n"
                "Up/Down recall previous inputs.\n"
                "Copying text:\n"
                "  - Drag to select, then Ctrl+Y to copy the selection\n"
                "  - /copy copies the full last answer\n"
                "  - Native Cmd+C/Ctrl+C: hold Option (mac) or Shift (Linux/Win)\n"
                "    while dragging to select in the terminal itself, then copy",
                style="grey62",
            ))
        elif name == "/clear":
            self.sess.new_chat()
            self.query_one("#plan-panel", PlanPanel).clear_plan()
            self.query_one("#chat-log", ChatLog).clear()
            self._log(Text("Started a new conversation.", style="grey50"))
        elif name == "/model":
            cfg = self.sess.runtime.llm_client.config
            if arg:
                cfg.default_model = arg
                self._log(Text(f"Model set to {arg} (this session).", style="grey50"))
            else:
                self._log(Text(f"Model: {cfg.default_model}", style="grey50"))
        elif name == "/effort":
            self._set_effort(arg)
        elif name == "/copy":
            self._copy_last_answer()
        elif name == "/export":
            self._export_last_answer(arg)
        elif name == "/config":
            if self._busy:
                self._log(Text("Finish the current query before /config.", style="grey50"))
            else:
                self.run_config()
        elif name == "/sessions":
            self._show_sessions()
        elif name == "/load":
            self._load_session(arg)
        elif name == "/roots":
            if self.sess.allowed_roots:
                self._log(Text("Local-search roots (granted via @path):", style="grey50"))
                for r in self.sess.allowed_roots:
                    self._log(Text(f"  {r}", style=ACCENT))
            else:
                self._log(Text("No local roots granted. Use @path to add one.", style="grey50"))
        elif name == "/verbose":
            self.verbose = not self.verbose
            self._log(Text(f"Verbose {'on' if self.verbose else 'off'}.", style="grey50"))
        else:
            self._log(Text(f"Unknown command: {name}. Try /help.", style="grey50"))

    def _set_effort(self, arg: str) -> None:
        cfg = self.sess.runtime.llm_client.config
        if not arg:
            current = cfg.reasoning_effort or "off"
            self._log(Text(f"Reasoning effort: {current}. Set with /effort <{'|'.join(EFFORT_LEVELS)}>.", style="grey50"))
            return
        level = arg.strip().lower()
        if level not in EFFORT_LEVELS:
            self._log(Text(f"Unknown effort '{arg}'. Choose: {', '.join(EFFORT_LEVELS)}.", style="grey50"))
            return
        cfg.reasoning_effort = "" if level == "off" else level
        self._log(Text(f"Reasoning effort set to {cfg.reasoning_effort or 'off'} (this session).", style="grey50"))

    def _copy_last_answer(self) -> None:
        if not self.sess.last_answer:
            self._log(Text("No answer to copy yet.", style="grey50"))
            return
        if _copy_to_clipboard(self.sess.last_answer):
            self._log(Text(f"Copied last answer ({len(self.sess.last_answer):,} chars) to clipboard.", style="grey50"))
        else:
            self._log(Text("No clipboard tool found (need xclip/xsel/wl-clipboard or pbcopy).", style="grey50"))

    def _export_last_answer(self, arg: str) -> None:
        if not self.sess.last_answer:
            self._log(Text("No answer to export yet.", style="grey50"))
            return
        path = arg.strip().strip('"').strip("'")
        if not path or not path.lower().endswith(".docx"):
            self._log(Text(
                "Please specify a filename ending in .docx, e.g. /export ~/report.docx",
                style="grey50",
            ))
            return
        try:
            from src.utils.docx_export import markdown_to_docx_bytes
            dest = Path(path).expanduser()
            docx_bytes = markdown_to_docx_bytes(self.sess.last_answer)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(docx_bytes)
        except Exception as exc:
            self._log(Text(f"Export failed: {exc}", style="grey50"))
        else:
            self._log(Text(f"Exported last answer to {dest}", style="grey50"))

    def action_copy_selection(self) -> None:
        """Copy the current drag-selection to the system clipboard (Ctrl+Y).

        Textual's mouse drag-selection highlights text but binds no copy key,
        so we wire one up. Falls back through OSC52 (Textual) and a local
        clipboard command so it works over SSH and locally.
        """
        text = self.screen.get_selected_text()
        if not text:
            self._log(Text("Nothing selected. Drag over text first, then Ctrl+Y.", style="grey50"))
            return
        # Textual's OSC52 path works over SSH; the local command path covers
        # terminals that strip OSC52. Try both.
        self.copy_to_clipboard(text)
        _copy_to_clipboard(text)
        self._log(Text(f"Copied selection ({len(text):,} chars).", style="grey50"))

    def _show_sessions(self) -> None:
        try:
            from src.utils.session_storage import SessionStorage
            rows = SessionStorage().list_sessions(limit=10)
        except Exception:
            rows = []
        # Remember the listing so /load <n> can resolve a number to a session.
        self._listed_sessions = rows
        if not rows:
            self._log(Text("No sessions found.", style="grey50"))
            return
        self._log(Text("Recent sessions:  (use /load <n> to continue one)", style="grey50"))
        for i, s in enumerate(rows, 1):
            ts = str(s.get("timestamp", ""))[:19]
            # Flatten newlines/runs of whitespace, and render the row as a
            # single non-wrapping line (overflow ellipsis) so a long query
            # never folds onto the next line and interleaves with timestamps.
            q = " ".join(str(s.get("query", "")).split())
            line = Text(no_wrap=True, overflow="ellipsis")
            line.append(f"  {i}. ", style=ACCENT)
            line.append(f"{ts}  ", style="grey50")
            line.append(q, style="white")
            self._log(line)

    def _load_session(self, arg: str) -> None:
        """Restore a past session's conversation (queries + answers only)."""
        rows = getattr(self, "_listed_sessions", None)
        if not rows:
            self._log(Text("Run /sessions first, then /load <n>.", style="grey50"))
            return
        if not arg.isdigit():
            self._log(Text("Usage: /load <number from /sessions>.", style="grey50"))
            return
        idx = int(arg) - 1
        if not (0 <= idx < len(rows)):
            self._log(Text(f"No session {arg}. Pick 1–{len(rows)}.", style="grey50"))
            return
        sid = rows[idx].get("session_id", "")
        try:
            from src.utils.session_storage import SessionStorage
            data = SessionStorage().load_session(sid)
        except Exception:
            data = None
        if not data:
            self._log(Text("Could not load that session.", style="grey50"))
            return
        turns = data.get("turns") or []
        pairs = self.sess.load_history(turns)
        if not pairs:
            self._log(Text("That session has no restorable conversation.", style="grey50"))
            return
        # Reset the visible UI and replay the restored conversation.
        self.query_one("#plan-panel", PlanPanel).clear_plan()
        self.query_one("#chat-log", ChatLog).clear()
        self._log(Text(f"Resumed session — {len(pairs)} turn(s) restored. Continue the thread below.", style="grey50"))
        self._log("")
        for q, a in pairs:
            self._log_user(q)
            if a:
                self._log_markdown(a)
            self._log("")

    @work(exclusive=True)
    async def run_config(self) -> None:
        """Re-run the setup wizard, then rebuild the runtime.

        The wizard uses plain input()/getpass, so we drop out of the
        full-screen TUI with App.suspend() while it runs, then restore.
        """
        cfg = cli_config.load_cli_config()
        try:
            with self.suspend():
                try:
                    cfg = cli_config.run_setup_wizard(cfg)
                except (EOFError, KeyboardInterrupt):
                    cfg = None
                if cfg is not None:
                    input("\n  Saved. Press Enter to return to SearchClaw… ")
        except Exception as e:  # e.g. SuspendNotSupported in odd terminals
            self._log(Text(f"Can't open the config wizard here: {e}", style="bold red"))
            return
        if cfg is None:
            self._log(Text("Config unchanged.", style="grey50"))
            return
        # Apply edited keys (force overwrite so a prior value doesn't shadow
        # them), then swap in a fresh runtime built from the new config.
        cli_config.apply_env_from_config(cfg, force=True)
        old = self.sess.runtime
        try:
            self.sess.runtime = build_runtime(cfg)
            await old.aclose()
            self._log(Text("Configuration reloaded.", style="grey50"))
            self._log(Text(f"model {self.sess.runtime.llm_client.config.default_model}", style="grey50"))
        except Exception as e:
            self._log(Text(f"Failed to reload config: {e}", style="bold red"))

    # --- chat log helpers -------------------------------------------

    def _log(self, renderable) -> None:
        self.query_one("#chat-log", ChatLog).write(renderable)

    def _log_markdown(self, md_text: str) -> None:
        """Mount a report/Markdown block as a Textual Markdown widget so its
        links are clickable (open in the browser via App.open_url)."""
        self.query_one("#chat-log", ChatLog).write(
            Markdown(md_text, open_links=True)
        )

    def _log_user(self, text: str) -> None:
        line = Text("❯ ", style=f"bold {SUCCESS}")
        line.append(text, style="bold white")
        self.query_one("#chat-log", ChatLog).write(line, classes="user-line")

    # --- exit: replay last report to the real terminal --------------

    def on_unmount(self) -> None:
        # Alt-screen clears on exit; echo the last answer so the user keeps it.
        if self.sess.last_answer:
            print("\n" + self.sess.last_answer + "\n")


def _copy_to_clipboard(text: str) -> bool:
    import shutil
    import subprocess
    import sys

    if sys.platform == "darwin":
        candidates = [["pbcopy"]]
    elif sys.platform == "win32":
        candidates = [["clip"]]
    else:
        candidates = [
            ["wl-copy"],
            ["xclip", "-selection", "clipboard"],
            ["xsel", "--clipboard", "--input"],
        ]
    for cmd in candidates:
        if shutil.which(cmd[0]) is None:
            continue
        try:
            subprocess.run(cmd, input=text.encode("utf-8"), check=True)
            return True
        except Exception:
            continue
    return False
