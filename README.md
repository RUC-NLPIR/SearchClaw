<p align="center">
  <img src="src/web/static/SearchClaw_icon.png" alt="SearchClaw" width="128" height="128" style="image-rendering: pixelated;">
</p>

<h1 align="center">SearchClaw</h1>

<h3 align="center">
  An agentic web research tool that searches, reads, and synthesizes well-cited answers.
</h3>

---

SearchClaw is a self-hosted research agent with a web UI. Give it a question and it autonomously searches the web, reads pages, checks academic papers and news, and produces a well-cited answer with source links. It runs as a local FastAPI server you can access from your browser.

For a detailed discussion of the system design, see the [Technical Report (PDF)](docs/technical_report.pdf).

## Features

- **Agentic research loop** -- autonomous multi-step research with search, fetch, read, and cite tools
- **Multiple search sources** -- web search (Google via Serper), academic papers (Semantic Scholar), news (NewsAPI / Google News RSS)
- **Quality gates** -- built-in hooks enforce citation minimums, source diversity, and answer completeness before finalizing
- **Research planning** -- automatic task decomposition for complex multi-part queries
- **Interactive clarification** -- the agent can ask you follow-up questions mid-research
- **Context compaction** -- automatic context window management for long research sessions
- **Persistent memory** -- learns from past sessions (source quality, user preferences, key facts)
- **Browser integration** -- optional Playwright/CDP support for JS-rendered pages and authenticated content
- **Password protection** -- optional password gate for remote deployments
- **Multi-provider LLM support** -- works with Anthropic, OpenAI, Google Gemini, and [many more](#supported-llm-providers) via [litellm](https://docs.litellm.ai/)

## Quick Start

### 1. Install

```bash
# Clone the repository
git clone https://github.com/DaoD/SearchClaw.git
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

## Configuration

All settings are in [`config/settings.yaml`](config/settings.yaml). The file is heavily commented; see it for full documentation.

Key sections:

| Section | What it controls |
|---------|-----------------|
| `llm` | Model selection, base URL for custom endpoints, retry settings |
| `limits` | Max agentic turns, context compaction threshold, rate limiting |
| `tools` | Search result counts, HTTP timeouts, content extraction |
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

## Architecture

```
src/
├── core/           # Agentic loop, tool registry, types
│   ├── loop.py     # Main research loop (stream events, tool calls, compaction)
│   ├── tool.py     # Tool base class and registry
│   ├── types.py    # Shared types (Message, ToolResult, Citation, etc.)
│   ├── context.py  # System prompt builder
│   └── compact.py  # Context window compaction
├── tools/          # Research tools the agent can use
│   ├── web_search.py       # Web search (Serper -> DuckDuckGo fallback)
│   ├── web_fetch.py        # Fetch & extract web pages (Jina -> direct fetch)
│   ├── academic_search.py  # Academic paper search (Semantic Scholar)
│   ├── news_search.py      # News search (NewsAPI -> Google News RSS)
│   ├── deep_read.py        # Read cached page sections
│   ├── cite_source.py      # Register citations for the answer
│   ├── research_plan.py    # Decompose complex queries into sub-tasks
│   └── ask_user.py         # Ask the user clarifying questions
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
