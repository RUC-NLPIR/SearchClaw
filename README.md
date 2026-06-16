<p align="center">
  <img src="src/web/static/SearchClaw_icon.png" alt="SearchClaw" width="128" height="128" style="image-rendering: pixelated;">
</p>

<h1 align="center">SearchClaw</h1>

<h3 align="center">
  An agentic web research tool that searches, reads, and synthesizes well-cited answers.
</h3>

---

SearchClaw is a self-hosted research agent with a web UI. Give it a question and it autonomously searches the web, reads pages, checks academic papers and news, and produces a well-cited answer with source links. It runs as a local FastAPI server you can access from your browser.

The design draws inspiration from **harness engineering** principles seen in tools like [Claude Code](https://docs.anthropic.com/en/docs/claude-code), where the scaffolding around the model is just as important as the model itself. Rather than relying on a single prompt to produce good answers, SearchClaw wraps the LLM in a structured harness: quality gate hooks reject answers that lack sufficient citations or source diversity, a research plan tool decomposes complex queries into trackable sub-tasks, two-phase context compaction keeps long sessions within the context window, each tool injects its own usage guidelines into the system prompt, and a persistent memory system carries learned facts and preferences across sessions. These mechanisms work together to make the agent more reliable and thorough than prompting alone would allow.

For a detailed discussion of the system design, see the [Technical Report (PDF)](report/SearchClaw.pdf).

https://github.com/user-attachments/assets/c9598751-da53-4e12-955d-870c9ff86b28

## Features

- **Agentic research loop** -- autonomous multi-step research with search, fetch, read, and cite tools
- **Multiple search sources** -- web search (Google via Serper), academic papers (Semantic Scholar & DBLP & arXiv), news (NewsAPI / Google News RSS), and wechat articles
- **Browser integration** -- optional Playwright/CDP support for JS-rendered pages and authenticated content
- **Interactive clarification** -- the agent can ask you follow-up questions mid-research
- **Research planning** -- automatic task decomposition for complex multi-part queries
- **On-demand skills** -- local `SKILL.md` instructions and optional Python scripts loaded only when relevant
- **Quality gates** -- built-in hooks enforce citation minimums, source diversity, and answer completeness before finalizing
- **Context compaction** -- automatic context window management for long research sessions
- **Persistent memory** -- learns from past sessions (source quality, user preferences, key facts)
- **Password protection** -- optional password gate for remote deployments
- **Multi-provider LLM support** -- works with Anthropic, OpenAI, Google Gemini, and [many more](#supported-llm-providers) via [litellm](https://docs.litellm.ai/)
- **Interactive CLI** -- a terminal client with the same research core plus local file search via `@path` mentions (see [Command-Line Interface](#command-line-interface))

## Benchmark Results

We evaluate SearchClaw on the first 50 questions of [BrowseComp](https://openai.com/index/browsecomp/) (OpenAI, 2025), a benchmark of 1,266 questions designed to test the ability of AI agents to browse the web and find hard-to-locate information.

| System | Model | Reasoning Effort | Accuracy | Avg Turns | Avg Searches | Avg Fetches | Avg Time | Refusals |
|--------|-------|-----------------|----------|-----------|-------------|-------------|----------|----------|
| ReAct baseline | GPT-5.4 (via Azure) | xhigh | 54.0% (27/50) | 46.3 | 37.9 | 27.5 | 9.5 min | 7 |
| **SearchClaw** | GPT-5.4 (via Azure) | xhigh | **74.0%** (37/50) | 40.2 | 45.5 | 26.2 | 10.2 min | 5 |

The ReAct baseline is a plain ReAct loop with the same `web_search` and `web_fetch` tools but no harness engineering — no quality hooks, research planning, content extraction, context compaction, or memory. SearchClaw's structured harness achieves a **+20 percentage point improvement** over the baseline.

**Experiment setup:**
- For fair comparison, SearchClaw disabled all search tools except `web_search` (i.e., `academic_search`, `news_search`, and `wechat_search` were removed), matching the ReAct baseline's tool set.
- Both systems were limited to a maximum of 50 search calls and 50 fetch calls per question. Once a limit is reached, the tool returns a dummy message prompting the agent to synthesize its final answer.
- To reduce API costs, both systems used self-hosted search and fetch services instead of the default Serper/Jina APIs.
- The GPT-5.4 service on Microsoft Azure exhibits occasional refusals due to its content filtering. All refused questions were retried once.

## Quick Start

### 1. Install

```bash
# Clone the repository
git clone https://github.com/RUC-NLPIR/SearchClaw.git
cd SearchClaw

# Install dependencies
pip install -e .

# Optional: browser integration
pip install -e '.[browser]'
playwright install chromium
```

**Requires Python 3.11+**

### 2. Set API Keys

```bash
# LLM provider (at least one required)
export ANTHROPIC_API_KEY="sk-ant-..."
# or
export OPENAI_API_KEY="sk-..."

# Web search (recommended -- falls back to DuckDuckGo scraping without this)
export SERPER_API_KEY="..."

# Web fetch (recommended -- falls back to direct HTTP without this)
export JINA_API_KEY="jina_..."

# News search (optional -- falls back to Google News RSS)
export NEWSAPI_KEY="..."
```

### 3. Run

```bash
python -m src.main
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

## Command-Line Interface

Besides the web UI, SearchClaw ships an interactive terminal client built on [Textual](https://textual.textualize.io/). It drives the same agentic research core as the server — same tools, hooks, planning, and memory — but runs entirely in your terminal, with one extra capability the web UI does not have: **local file search**.

For a full walkthrough with screenshots, see the [CLI guide](https://daod.github.io/project/searchclaw).

### Launch

Installing the package (`pip install -e .`) registers a `searchclaw` console command:

```bash
searchclaw
```

On first run it walks you through a short setup wizard (LLM endpoint, models, API keys) and stores the result at `~/.searchclaw/config.yaml`. Subsequent launches go straight to the chat screen. Type a question and press Enter; the agent streams its answer live, with citations, in the terminal.

### Slash commands

Type `/` to see inline command suggestions. Available commands:

| Command | Description |
|---------|-------------|
| `/help` | Show all commands and key bindings |
| `/clear` | Start a new conversation |
| `/stop` | Interrupt the current research turn (keeps what was streamed so far) |
| `/config` | Re-run the setup wizard (endpoint, models, keys) |
| `/model [name]` | Show or set the model for this session |
| `/effort [level]` | Show or set reasoning effort: `off`/`low`/`medium`/`high`/`xhigh`/`max` |
| `/copy` | Copy the last answer (raw markdown) to the clipboard |
| `/export <path.docx>` | Export the last answer to a DOCX file |
| `/roots` | Show local-search directories granted via `@path` |
| `/sessions` | List recent saved sessions |
| `/load <n>` | Resume a past session (after `/sessions`) |
| `/verbose` | Toggle reasoning output |
| `/exit` | Quit |

Use **Up/Down** to recall previous inputs. Drag to select text, then **Ctrl+Y** to copy the selection.

### Local file search with `@path`

In the CLI you can point the agent at files on your own machine by prefixing a path with `@`. This grants the agent read-only access to that directory (or file) for the rest of the session, and the agent can then search and read those files alongside the web:

```text
@~/papers what do these say about retrieval-augmented generation?
@./report.pdf summarize the key findings
```

A directory mention grants the whole directory; a file mention grants its parent and points the agent at that file specifically. Three local tools become available once a root is granted:

- **`local_glob`** — list files/dirs by name (sees PDFs and Office documents)
- **`local_search`** — grep file contents for a text pattern (plain-text files)
- **`local_read`** — read a file or line range, and extract text from PDF/`.docx`/`.pptx`

Access is sandboxed to the granted roots: path traversal and symlink escapes are blocked, and noise directories (`.git`, `node_modules`, …), secret files (`.env`, SSH keys, …), and oversized/binary files are skipped automatically. Use `/roots` to see what is currently granted.

You can mention paths containing spaces by quoting them (`@"~/my docs"`), and non-ASCII paths — including Chinese, Japanese, and Korean — are supported even when the path runs directly into the following text with no space (e.g. `@~/论文目录里有哪些文章`).

## Configuration

All settings are in [`config/settings.yaml`](config/settings.yaml). The file is heavily commented; see it for full documentation.

Key sections:

| Section | What it controls |
|---------|-----------------|
| `llm` | Model selection, base URL for custom endpoints, retry settings |
| `limits` | Max agentic turns, context compaction threshold, rate limiting |
| `tools` | Search result counts, HTTP timeouts, content extraction |
| `skills` | On-demand local skills and bundled skill script limits |
| `hooks` | Quality gates (min citations, min domains, min answer length) |
| `browser` | Playwright/CDP browser integration |
| `memory` | Persistent memory system |
| `server` | Host, port, API key, CORS |

### Changing the LLM

SearchClaw uses [litellm](https://docs.litellm.ai/) for LLM routing. Model names use a `provider/model` format:

```yaml
llm:
  default_model: "anthropic/claude-sonnet-4-20250514"
  side_query_model: "anthropic/claude-haiku-3-20250305"
```

To use OpenAI:
```yaml
llm:
  default_model: "openai/gpt-4o"
  side_query_model: "openai/gpt-4o-mini"
```

To use a local/custom endpoint (vLLM, Ollama, LiteLLM proxy):
```yaml
llm:
  default_model: "openai/my-model-name"
  base_url: "http://localhost:8000/v1"
```

### Password Protection

For remote deployments, set a password to protect the UI:

```yaml
server:
  api_key: "your-secret-password"
```

Or via environment variable:
```bash
export SEARCH_CLAW_API_KEY="your-secret-password"
```

## Skills

SearchClaw supports on-demand local skills for the main web/API system. A skill is a folder containing a `SKILL.md` file with metadata and task-specific instructions. At startup, SearchClaw discovers available skills and shows the model only their names and summaries. The full skill body is loaded only when the model calls the `use_skill` tool.

Default skill directory:

```text
skills/<skill-name>/SKILL.md
```

Minimal skill example:

```markdown
---
name: evidence-ledger
description: Maintain a structured evidence ledger for multi-source research tasks.
when_to_use: Use when an answer requires careful source tracking, conflict checks, or evidence synthesis.
---

# Evidence Ledger

Follow this workflow when gathering and comparing evidence...
```

Optional skill scripts may live inside the same skill directory and can be invoked with `run_skill_script` after the skill is loaded:

```text
skills/<skill-name>/scripts/analyze.py
```

Script execution is intentionally restricted: only Python `.py` files inside the selected skill directory can run, no shell is used, and arguments are passed as a JSON array of strings.

Configure skills in `config/settings.yaml`:

```yaml
skills:
  enabled: true
  dirs: ["./skills"]
  listing_max_chars: 8000
  max_skill_chars: 50000
  script_timeout_seconds: 30
  script_max_output_chars: 20000
```

The benchmark, baseline, and judge paths do not load skills.

## Architecture

```
src/
├── core/           # Agentic loop, tool registry, types
│   ├── loop.py     # Main research loop (stream events, tool calls, compaction)
│   ├── tool.py     # Tool base class and registry
│   ├── types.py    # Shared types (Message, ToolResult, Citation, etc.)
│   ├── context.py  # System prompt builder
│   └── compact.py  # Context window compaction
├── skills/         # Local SKILL.md discovery and metadata parsing
│   └── loader.py   # Skill loader used by the main web/API system
├── tools/          # Research tools the agent can use
│   ├── web_search.py       # Web search (Serper -> DuckDuckGo fallback)
│   ├── web_fetch.py        # Fetch & extract web pages (Jina -> direct fetch)
│   ├── academic_search.py  # Academic paper search (Semantic Scholar)
│   ├── news_search.py      # News search (NewsAPI -> Google News RSS)
│   ├── wechat_search.py    # WeChat article search
│   ├── deep_read.py        # Read cached page sections
│   ├── cite_source.py      # Register citations for the answer
│   ├── research_plan.py    # Decompose complex queries into sub-tasks
│   ├── ask_user.py         # Ask the user clarifying questions
│   ├── use_skill.py        # Load full local skill instructions on demand
│   └── run_skill_script.py # Run Python scripts bundled inside loaded skills
├── hooks/          # Quality gates that run before finalizing answers
│   ├── engine.py           # Hook execution engine
│   ├── builtin_hooks.py    # Citation, diversity, and completeness checks
│   └── plan_completeness_hook.py  # Ensure all sub-tasks are done
├── memory/         # Persistent cross-session memory
│   ├── store.py            # File-based memory storage
│   ├── retrieval.py        # Relevance-based memory retrieval
│   └── extract.py          # Post-session memory extraction
├── llm/            # LLM client (litellm wrapper)
│   └── client.py           # Streaming LLM calls with retry
├── utils/          # Shared utilities
│   ├── browser_manager.py  # Playwright/CDP browser lifecycle
│   ├── browser_search.py   # Browser-based search fallback
│   ├── browser_fetch.py    # Browser-based page fetch
│   ├── content_extractor.py # LLM-based content compression
│   ├── html_to_markdown.py # HTML -> markdown conversion
│   ├── rate_limiter.py     # Per-domain rate limiting
│   ├── session_storage.py  # Session persistence (JSON files)
│   ├── token_counter.py    # Token counting (tiktoken)
│   └── url_validator.py    # URL validation and SSRF protection
├── web/            # FastAPI server and web UI
│   ├── router.py           # API routes, WebSocket handler, auth middleware
│   └── static/             # Frontend (vanilla HTML/CSS/JS)
└── main.py         # Entry point
```

### How It Works

1. You type a question in the web UI
2. The question is sent to the server via WebSocket
3. The **agentic loop** starts: the LLM reads the question and the available tools
4. The LLM decides which tools to call (search, fetch, cite, etc.), and tool calls run in parallel when possible
5. Tool results are fed back to the LLM, which decides the next step
6. When the LLM produces a final answer, **quality hooks** check it:
   - Enough citations? Diverse sources? Sufficient detail?
   - If not, the agent is told to keep researching
7. The final answer with citations is streamed to the UI

## Supported LLM Providers

SearchClaw supports any provider that litellm supports. Tested and commonly used providers:

| Provider | Prefix | Example model | API key env var |
|----------|--------|---------------|-----------------|
| Anthropic | `anthropic/` | `anthropic/claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai/` | `openai/gpt-4o` | `OPENAI_API_KEY` |
| Google Gemini | `gemini/` | `gemini/gemini-2.0-flash` | `GEMINI_API_KEY` |
| xAI (Grok) | `xai/` | `xai/grok-2-latest` | `XAI_API_KEY` |
| Qwen (Alibaba) | `dashscope/` | `dashscope/qwen-plus` | `DASHSCOPE_API_KEY` |
| Doubao (ByteDance) | `volcano/` | `volcano/doubao-pro-32k` | `VOLC_API_KEY` |
| Minimax | `minimax/` | `minimax/MiniMax-Text-01` | `MINIMAX_API_KEY` |
| GLM (Zhipu AI) | `zhipuai/` | `zhipuai/glm-4-plus` | `ZHIPUAI_API_KEY` |
| Moonshot (Kimi) | Use `openai/` prefix | `openai/moonshot-v1-auto` | `OPENAI_API_KEY` + `base_url` |

For providers without a dedicated litellm prefix (Kimi, Mimo, etc.), use the `openai/` prefix with a custom `base_url` pointing to their OpenAI-compatible endpoint.

See [litellm docs](https://docs.litellm.ai/docs/providers) for the full provider list.

## Browser Integration

For JS-heavy pages, paywalled content, or authenticated sites, enable browser integration:

```yaml
browser:
  enabled: true
  mode: "playwright"    # or "cdp" for your real Chrome
```

**Playwright mode** (default): launches a managed Chromium instance. Good for JS-rendered pages.

**CDP mode**: connects to your running Chrome via DevTools Protocol, inheriting your cookies, extensions, and logins. Best for authenticated content (Twitter, LinkedIn, etc.).

```bash
# Launch Chrome with CDP enabled
google-chrome --remote-debugging-port=9222 --user-data-dir="$HOME/chrome-cdp-profile"
```

## License

MIT
