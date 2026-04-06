# Search Agent: System Design Technical Report

## Abstract

This report presents the system design of a web research agent that autonomously searches, retrieves, and synthesizes information from the internet to answer complex user queries. The system is built around a single agentic loop architecture inspired by Claude Code's `query.ts` pattern, augmented with domain-specific mechanisms for content compression, persistent memory, structured research planning, and quality-gated answer finalization. The agent operates as a streaming WebSocket service, enabling real-time research progress visibility. Key contributions include a pre-extraction content compression strategy that reduces context window consumption by 70-90% compared to raw content injection, a three-tier URL filtering system for intelligent content distillation, and a hook-based quality gate system that enforces citation diversity and research plan completeness before answer finalization.

## 1. Introduction

Large language models (LLMs) demonstrate strong reasoning capabilities but are limited by static training data and a tendency to hallucinate when asked about current events, niche topics, or questions requiring precise factual accuracy. A web research agent addresses these limitations by equipping an LLM with tools to search the web, read pages, and cite sources — grounding its responses in verifiable, up-to-date evidence.

Our search agent is designed around three core principles:

1. **Accuracy over speed** — every factual claim must be backed by cited sources from diverse domains.
2. **Comprehensive coverage** — complex queries are decomposed into sub-tasks to prevent shallow treatment of any dimension.
3. **Context efficiency** — web content is compressed before entering the main context window, preserving research capacity across many sources.

The system architecture draws inspiration from Claude Code (Anthropic's agentic coding tool), adapting its single-loop, tool-calling, and hook-based patterns for the web research domain. However, several design decisions diverge from Claude Code where the research agent's requirements differ fundamentally — particularly in content handling, where pre-extraction compression replaces Claude Code's post-hoc compaction approach.

### 1.1 System Overview

```
                        ┌─────────────────────────────┐
                        │       Frontend (Web UI)       │
                        │   WebSocket + Streaming UI    │
                        └─────────────┬───────────────┘
                                      │ StreamEvents
                        ┌─────────────▼───────────────┐
                        │     Web Router (FastAPI)      │
                        │  Session management, auth,    │
                        │  memory retrieval, config     │
                        └─────────────┬───────────────┘
                                      │
                ┌─────────────────────▼─────────────────────┐
                │              Agentic Loop                  │
                │  while(true): LLM → tools → hooks → ...  │
                │                                            │
                │  ┌─────────┐ ┌──────────┐ ┌───────────┐  │
                │  │ Context  │ │Compaction│ │  Quality   │  │
                │  │ Builder  │ │ Engine   │ │  Hooks     │  │
                │  └─────────┘ └──────────┘ └───────────┘  │
                └──────┬────────────┬───────────┬───────────┘
                       │            │           │
        ┌──────────────▼──────┐     │     ┌─────▼──────────┐
        │    Tool Registry     │     │     │  Memory System  │
        │  ┌────────────────┐  │     │     │  ┌──────────┐  │
        │  │  web_search    │  │     │     │  │  Store    │  │
        │  │  web_fetch     │  │     │     │  │  Extract  │  │
        │  │  deep_read     │  │     │     │  │  Retrieve │  │
        │  │  cite_source   │  │     │     │  └──────────┘  │
        │  │  research_plan │  │     │     └────────────────┘
        │  │  ask_user      │  │     │
        │  │  academic_*    │  │     │
        │  │  news_*        │  │     │
        │  └────────────────┘  │     │
        └──────────────────────┘     │
                                     │
                        ┌────────────▼────────────┐
                        │      LLM Client          │
                        │  litellm (streaming)     │
                        │  + side_query (cheap)    │
                        └──────────────────────────┘
```

## 2. Architecture

### 2.1 Single Agentic Loop

The system's core is a single `while(True)` loop implemented as a Python `AsyncGenerator` (`src/core/loop.py`). This design mirrors Claude Code's `query.ts` rather than a multi-agent architecture. The rationale: a web research session is an inherently sequential reasoning chain — search, evaluate, read, synthesize — where each step informs the next. Multi-agent patterns (parallel specialized agents, orchestrator-worker) add coordination overhead without meaningful benefit for this workflow.

The loop follows a six-phase iteration cycle:

```
┌─────────────────────────────────────────────────────┐
│                  Loop Iteration                      │
│                                                      │
│  1. Guard Check ──→ max_turns exceeded? → break     │
│         │                                            │
│  2. Compaction ──→ context too large? → compress     │
│         │                                            │
│  3. LLM Call ──→ stream response + tool calls       │
│         │                                            │
│  4. Tool Calls? ──→ No → Stop Hooks → break/loop    │
│         │ Yes                                        │
│  5. Execute Tools ──→ parallel if safe              │
│         │                                            │
│  6. Inject Results ──→ citations, plan updates      │
│         └──→ continue to step 1                     │
└─────────────────────────────────────────────────────┘
```

**Key design decisions:**

- **Bidirectional AsyncGenerator**: The loop uses `yield` to emit `StreamEvent` objects and `asend()` to receive user answers for interactive tools (the `ask_user` tool). This avoids Futures, deadlocks, and background tasks — the loop and the WebSocket handler communicate through a single coroutine channel.

- **Explicit `LoopState`**: All mutable state — messages, citations, turn count, compaction count, research plan — is carried in a `LoopState` dataclass. The loop destructures this at each iteration entry point.

- **Parallel tool execution**: Tools declaring `is_concurrency_safe = True` (e.g., `web_search`, `web_fetch`) execute concurrently via `asyncio.gather()`. State-mutating tools (e.g., `research_plan`) run sequentially.

- **LLM error isolation**: If the LLM call fails, the loop breaks immediately without running stop hooks. This prevents a death spiral where hooks evaluating an empty response generate feedback that triggers another failed LLM call.

### 2.2 Streaming Event Protocol

The loop communicates with the presentation layer through a typed event protocol (`StreamEvent`), fully decoupling the research logic from the UI. Events include:

| Event Type | Data | Description |
|---|---|---|
| `TEXT_DELTA` | `{text: str}` | Streaming text token from the LLM |
| `TOOL_USE` | `{tool_name, tool_input, tool_use_id}` | LLM requested a tool call |
| `TOOL_RESULT` | `{tool_name, result, is_error, truncated}` | Tool execution completed |
| `CITATION` | `{url, title, snippet, source_type}` | New citation discovered |
| `STATUS` | `{message: str}` | Status update (compacting, executing tools) |
| `PLAN_UPDATE` | `{tasks[], completed_count, ...}` | Research plan state changed |
| `USER_QUESTION` | `{question, options[]}` | Interactive clarification needed |
| `ERROR` | `{message: str}` | Error occurred |
| `DONE` | `{citations[], session_summary, ...}` | Research complete |

### 2.3 Tool Architecture

#### 2.3.1 Tool Base Class

All tools extend an abstract `Tool` base class (`src/core/tool.py`) that defines a common interface:

```python
class Tool(ABC):
    name: str                          # Unique identifier
    description: str                   # Used by LLM for tool selection
    input_schema: dict                 # JSON Schema for function calling
    is_concurrency_safe: bool = False  # Can run in parallel?
    is_read_only: bool = True          # No side effects?
    max_result_size_chars: int = 50_000  # Context blowup prevention

    @abstractmethod
    async def call(self, args: dict, context: ToolUseContext) -> ToolResult

    def prompt(self) -> str: ...       # System prompt contribution
    def validate_input(self, args: dict) -> ValidationResult: ...
    def to_api_schema(self) -> dict: ... # OpenAI function calling format
```

**Design principles** (mirroring Claude Code):
- **Fail-closed defaults**: `is_concurrency_safe=False` prevents accidental parallel mutation. New tools must explicitly opt in to concurrency.
- **Per-tool prompt contribution**: Each tool's `prompt()` method is automatically appended to the system prompt, so adding a new tool requires no changes to the prompt assembly logic.
- **Input validation before execution**: `validate_input()` catches malformed inputs (e.g., invalid URLs, empty queries) before the tool runs, preventing wasted API calls.
- **Result size limits**: `max_result_size_chars` prevents individual tool results from consuming the entire context window.

#### 2.3.2 Tool Registry

The `ToolRegistry` (`src/core/tool.py`) collects all tools at startup and provides lookup by name. It also generates API schemas for the LLM's function calling interface and identifies which tools are concurrency-safe for parallel execution.

```python
registry = build_default_registry(config)
# Core tools: web_search, web_fetch, deep_read, cite_source,
#             research_plan, ask_user
# Optional:   academic_search, news_search (if dependencies available)
```

#### 2.3.3 Tool Inventory

The system provides eight tools organized into four functional categories:

**Discovery tools** — find relevant sources:
- `web_search`: Searches via Serper.dev API (primary) with DuckDuckGo HTML fallback. Returns titles, URLs, and snippets. Concurrency-safe.
- `academic_search`: Searches academic papers (optional, depends on external API). Returns structured paper metadata.
- `news_search`: Searches recent news with configurable time window (default: 7 days back). Optional.

**Content retrieval tools** — read and extract information:
- `web_fetch`: Fetches a URL and converts to markdown. Dual-strategy: Jina Reader API (handles JS, PDFs) with direct HTTP + trafilatura fallback. Large pages trigger content extraction. Concurrency-safe.
- `deep_read`: Reads specific sections from cached page content. Supports heading-based extraction, keyword search, and line ranges. Includes path traversal protection (only reads from the cache directory).

**Research management tools** — structure the research process:
- `research_plan`: Creates and tracks structured sub-tasks (create/update/check actions). State lives on `LoopState.research_plan`. Not concurrency-safe (mutates shared state).
- `ask_user`: Pauses research to ask the user a clarifying question with selectable options. Uses the bidirectional generator pattern for interaction.

**Citation tools** — formal source registration:
- `cite_source`: Registers a citation with URL, title, snippet, source type, and relevance note. Citations with `cited=True` are distinguished from automatically-discovered citations in the final output.


## 3. Content Retrieval and Compression

### 3.1 The Content Pipeline

When the agent fetches a web page, the content passes through a multi-stage pipeline before entering the main context window:

```
URL
 │
 ▼
┌──────────────────┐
│  1. Fetch         │  Jina Reader API (JS/PDF support)
│     Strategy      │  or direct HTTP + trafilatura
└────────┬─────────┘
         │ raw markdown (potentially 100K+ chars)
         ▼
┌──────────────────┐
│  2. Size Check    │  content ≤ threshold? → return raw
│     (15K default) │  content > threshold? → continue ▼
└────────┬─────────┘
         │ large content
         ▼
┌──────────────────┐
│  3. Cache to Disk │  Full content saved for deep_read
│     (always)      │  e.g., cache/web_fetch_a3f2c1.md
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  4. LLM Extract  │  side_query extracts:
│     (cheap model) │  - Key facts & data points
│                   │  - Relevant URLs (filtered)
│                   │  - Structural URLs (always kept)
│                   │  - Notable quotes (1-3)
└────────┬─────────┘
         │ ~3-5K chars (vs 50K+ raw)
         ▼
┌──────────────────┐
│  5. Return to     │  Extracted content + reference
│     Main Context  │  to cached full content
└──────────────────┘
```

### 3.2 Pre-Extraction Content Compression

The key insight driving our content compression design is that for a research agent doing synthesis across multiple pages, what matters is **facts and attribution**, not raw page content. Unlike Claude Code, which adds raw tool results to the context and compacts later, our system compresses content **before** it enters the main context window.

When fetched content exceeds the extraction threshold (default: 15,000 characters, configurable via `content_extraction_threshold` in `settings.yaml`), a cheap side-query LLM call (`src/utils/content_extractor.py`) extracts:

1. **Key facts**: All factual claims, data points, statistics, dates, and findings relevant to the research question. Precision is preserved — numbers, percentages, dates, and proper nouns are retained.

2. **Relevant URLs**: References, cited studies, linked primary sources — only URLs a researcher would want to follow up on. Filtered by relevance to the research question.

3. **Structural URLs**: Pagination links, "read more" links, download links, and links to other sections of the same document. These are **always kept** regardless of topic relevance, as they enable navigation to additional content.

4. **Notable quotes**: 1-3 direct quotes worth citing, with attribution.

**What is discarded**: Navigation menus, social media share links, ad links, cookie/privacy policy links, "about us" or "contact" pages, login/signup links, sidebar promotions, and content unrelated to the research question.

The extraction prompt is query-aware — it receives the user's original research question (propagated via `ToolUseContext.extra["research_query"]`) and uses it for relevance filtering. Content below the threshold is returned raw (no extraction overhead).

**Compression ratio**: Typical extraction reduces 50,000+ characters of raw web content to ~3,000-5,000 characters of dense findings, achieving a 70-90% reduction in context consumption per page while preserving all research-relevant information.

**Fallback**: If the side-query LLM call fails, the system falls back to raw truncation at `max_result_size_chars` (default: 50,000 characters), the same behavior as before extraction was implemented.

### 3.3 The Deep Read Pattern

Full page content is always cached to disk before extraction, enabling a two-tier access pattern:

- **Fast path** (extraction): Compressed key facts enter the main context immediately.
- **Deep path** (`deep_read` tool): The agent can retrieve specific sections from the cached full content by specifying a heading name, keywords, or line range.

This is analogous to how a human researcher would skim an article for key points (extraction), then go back to read a specific section in detail (deep_read) when needed.

The `deep_read` tool includes **path traversal protection** — it verifies that the requested `cached_path` is within the configured cache directory, preventing the LLM from being tricked into reading arbitrary filesystem paths (e.g., `/etc/passwd`, `~/.ssh/id_rsa`).

### 3.4 Context Compaction

Beyond pre-extraction, the system includes a post-hoc compaction mechanism (`src/core/compact.py`) that activates when the total context size exceeds a configurable threshold (default: 80,000 tokens). This is a safety net for sessions that accumulate many tool results even after extraction.

**Two-phase compaction:**

1. **Microcompact**: Preserves the first user message (original query) and the last 4 messages in full. All older tool result messages are truncated to 300 characters; older assistant messages are truncated to 500 characters. This is fast, requires no LLM call, and is often sufficient.

2. **Full compact**: If microcompact doesn't reduce the context enough, a side-query LLM call summarizes the entire conversation. The summary preserves:
   - The original research question
   - All factual findings discovered
   - All source URLs (critical for citations)
   - Unresolved questions and leads
   - Contradictions between sources

   The compacted message list becomes: `[original_query, summary_message, last_assistant_message]`.

**Full compact fallback**: If the side-query fails, the system falls back to aggressive microcompact (keeping only the last 2 messages instead of 4).


## 4. Research Planning System

### 4.1 Motivation

Without structured planning, LLMs exhibit a common failure mode on complex queries: they research the first aspect thoroughly, then generate a partial answer without investigating remaining dimensions. For example, given "Compare the economic policies of the US, EU, and China on AI regulation," an unguided agent might deeply research the US position but give only a sentence each to the EU and China.

### 4.2 Research Plan Data Model

The research plan (`src/core/types.py`) consists of two dataclasses:

```python
@dataclass
class ResearchTask:
    id: str                  # Auto-assigned: "1", "2", ...
    title: str               # Short description
    details: str = ""        # Strategy/context
    status: Literal["pending", "in_progress", "completed"] = "pending"
    findings: str = ""       # Recorded results

@dataclass
class ResearchPlan:
    tasks: list[ResearchTask]
```

The `ResearchPlan` provides:
- `get_task(id)`: O(n) lookup by task ID
- `completed_count`: Count of completed tasks
- `is_complete`: True iff all tasks are completed (empty plans are never complete)
- `summary()`: Human-readable progress with status icons (`○` pending, `◉` in progress, `●` completed)
- `to_dict()`: Serialization for WebSocket streaming

The plan lives on `LoopState.research_plan` — entirely in-memory, single-session, no file persistence.

### 4.3 The Research Plan Tool

The `research_plan` tool (`src/tools/research_plan.py`) exposes three actions through a single tool interface:

| Action | Purpose | Parameters |
|---|---|---|
| `create` | Decompose query into 3-7 sub-tasks | `tasks: [{title, details?}]` |
| `update` | Record progress and findings | `task_id, status, findings?` |
| `check` | View current plan status | (none) |

**Design decisions:**

- **Single tool, multiple actions**: Keeps the tool namespace small and reduces prompt overhead vs. three separate tools.
- **One-way status progression**: The LLM can only set tasks to `in_progress` or `completed`, never back to `pending`. There is no "delete task" action.
- **Overwrite-on-create**: Creating a new plan replaces any existing plan. No merge or append logic. One plan per session.
- **Not concurrency-safe**: The tool mutates `LoopState` directly, so it must run sequentially.

### 4.4 Plan Enforcement Mechanisms

Two mechanisms ensure the agent follows through on its research plan:

#### 4.4.1 Soft Nudge

After each tool execution cycle, the loop checks whether the agent has performed 3 or more searches without creating a research plan. If so, it injects a synthetic user message nudging the LLM to create one:

```
"You've done several searches without creating a research plan.
 This query appears to have multiple aspects. Please use
 research_plan(action='create') now to organize your remaining
 research into sub-tasks before continuing."
```

This nudge fires at most once per session (tracked via the `_tag: "plan_nudge"` metadata marker on the injected message).

#### 4.4.2 Plan Completeness Hook

The `PlanCompletenessHook` (`src/hooks/plan_completeness_hook.py`) is a stop hook that prevents the agent from finalizing its answer while sub-tasks remain incomplete. When the LLM produces a response without tool calls (indicating it wants to finalize):

| Condition | Result |
|---|---|
| No plan exists | Pass (no constraint) |
| All tasks completed | Pass |
| Incomplete tasks remain | **Block** — inject feedback listing remaining tasks |

The blocking feedback explicitly instructs the agent to continue researching the remaining topics, creating a forced loop continuation that ensures comprehensive coverage.

### 4.5 Plan Findings and Memory Integration

At session finalization, the loop extracts all non-empty `findings` from completed tasks and assembles them into a structured summary:

```python
plan_findings = "\n".join(
    f"- {t.title}: {t.findings}"
    for t in state.research_plan.tasks
    if t.findings
)
```

This `plan_findings` string is passed to the post-session memory extraction system, providing structured context about what the agent learned — enabling more precise memory formation than raw conversation analysis.


## 5. Quality Gate System (Hooks)

### 5.1 Hook Architecture

The hook system (`src/hooks/engine.py`) implements quality checks as gates that run at key lifecycle points. The primary hook event is `stop` — triggered when the LLM attempts to finalize its answer (produces a response without tool calls).

```python
class Hook(ABC):
    name: str
    description: str

    @abstractmethod
    async def evaluate(self, state: LoopState, **kwargs) -> HookEvaluation:
        """Returns (passed: bool, feedback: str)"""

class HookEngine:
    stop_hooks: list[Hook]

    async def run_stop_hooks(self, state: LoopState) -> HookResult:
        """First failure stops evaluation and returns feedback"""
```

When any stop hook fails, its `feedback` string is injected into the conversation as a user message, forcing the agent to continue researching. The loop continues until all hooks pass or `max_turns` is reached.

**Fail-open on errors**: If a hook's `evaluate()` method raises an exception, it is skipped (logged as a warning). This prevents hook bugs from blocking answer delivery.

### 5.2 Built-in Quality Hooks

Four built-in hooks enforce minimum quality standards:

#### CitationQualityHook
Requires at least `min_citations` (default: 2) explicitly registered citations before finalizing. **Memory-aware**: if the agent answered without using any tools (likely drawing on injected memory content), the feedback is targeted — "verify and cite your claims" rather than a generic "search more" — avoiding wasted turns re-researching known topics.

#### SourceDiversityHook
Requires citations from at least `min_domains` (default: 2) distinct website domains. Prevents the agent from citing a single source repeatedly. Domain normalization strips "www." prefixes.

#### AnswerCompletenessHook
Requires the final answer to be at least `min_answer_chars` (default: 200) characters long. Catches cop-out responses like "I couldn't find information about this."

#### PlanCompletenessHook
Blocks finalization while research plan sub-tasks remain incomplete (detailed in Section 4.4.2).

### 5.3 Non-Research Query Detection

All citation-related hooks share a `_needs_research()` classifier that determines whether the current query requires web research at all. Queries like "Hello", "What can you do?", or "Thanks!" are classified as non-research and exempted from citation requirements.

The classifier uses a **cheap LLM side-query** (not regex patterns) for generalizability:

```python
async def _needs_research(state: LoopState) -> bool:
    # Fast path: if any tool was used, research was needed
    if any(m.role == "tool" for m in state.messages):
        return True

    # Side-query classification with structured output
    response = await side_query(
        prompt=f'User query: "{first_user_msg}"',
        system="Classify whether this query requires web research...",
        output_schema={"type": "object", "properties": {"needs_research": {"type": "boolean"}}, ...},
    )
```

Results are cached per-state to avoid redundant side-query calls when multiple hooks check the same state in a single stop-hook cycle. The cache is cleared after each evaluation cycle to prevent unbounded memory growth.


## 6. Persistent Memory System

### 6.1 Overview

The memory system (`src/memory/`) enables the agent to learn from past sessions — remembering user preferences, source reputation, quality feedback, and useful references. This mirrors Claude Code's `memdir/` system.

### 6.2 Memory Types

Four categories of memories are defined (`src/memory/types.py`):

| Type | Description | Example |
|---|---|---|
| `user` | User background, expertise, preferences | "User is a PhD student in NLP" |
| `feedback` | Corrections on search behavior | "User prefers primary sources over news articles" |
| `source_reputation` | Trusted/untrusted sources | "arxiv.org is reliable for ML papers" |
| `reference` | Bookmarked sources, useful URLs | "WHO COVID dashboard: https://..." |

### 6.3 Storage Format

Memories are stored as individual markdown files with YAML frontmatter in a configurable directory (default: `./memory/`):

```markdown
---
title: User prefers academic sources
type: feedback
tags: [preferences, sources]
created: 2024-03-15T10:30:00
updated: 2024-03-15T10:30:00
---

The user has indicated they prefer peer-reviewed academic papers
over news articles or blog posts when available.
```

**Naming conventions:**
- `user` type: Always `user_profile.md` (singleton, overwritten on update)
- Other types: `{type}_{sanitized_title}_{timestamp}.md`

A `MEMORY.md` index file is automatically maintained as a compact overview of all stored memories, grouped by type. This index is designed to fit within a few hundred tokens.

### 6.4 Memory Retrieval (Pre-Query)

Before each research session, the system retrieves relevant memories to inject into the system prompt (`src/memory/retrieval.py`). The retrieval strategy is LLM-based, not keyword-based:

1. **Load headers**: Read frontmatter metadata from all memory files (lightweight, no full content loading).
2. **LLM selection**: A side-query with structured JSON output selects the most relevant memory titles for the current query. The selection prompt explicitly allows returning an empty list.
3. **Load selected**: Full content is loaded only for the selected memories (up to `max_relevant_memories`, default: 5).
4. **Format for prompt**: Selected memories are formatted and injected into the system prompt's `## Memories` section.

**Design principles (mirroring Claude Code's `findRelevantMemories.ts`):**
- **Always run the selector**, even when memory count ≤ max_memories. This prevents loading irrelevant memories.
- **Allow empty results**: The system prompt and selection prompt both explicitly permit "no relevant memories."
- **Fail to empty**: On error, return `[]` rather than falling back to recent memories. Never inject potentially irrelevant content.

### 6.5 Memory Extraction (Post-Session)

After each completed research session, an extraction process (`src/memory/extract.py`) analyzes the conversation and saves 0-3 noteworthy learnings:

```python
async def extract_memories(
    query: str,           # Original user query
    final_answer: str,    # Agent's final response
    plan_findings: str,   # Research plan findings (if any)
    store: MemoryStore,
) -> list[MemoryEntry]:
```

The extraction is:
- **LLM-driven**: A side-query analyzes the session summary and identifies memories worth saving.
- **Duplicate-aware**: Existing memory headers are included in the extraction prompt to prevent duplicate entries.
- **Capped**: Maximum 3 memories per session.
- **Fire-and-forget**: Runs after the WebSocket response is complete, never blocking the user.
- **Fail-safe**: Extraction failures are logged but never surface to the user.

### 6.6 Memory in Context

When memories are injected into the system prompt, they come with an explicit warning:

> *IMPORTANT: Memories are context hints from past sessions, NOT citable sources. You MUST still search and verify information even if memories seem to answer the question. Never use memories as your sole basis for an answer — always search for up-to-date sources and cite them.*

This prevents the agent from short-circuiting research by citing memory content as fact. The `CitationQualityHook` reinforces this — if the agent answers using only memory (no tool calls), it receives targeted feedback: "verify and cite your claims."


## 7. System Prompt Assembly

### 7.1 Architecture

The system prompt is assembled by `ContextBuilder` (`src/core/context.py`), which concatenates six sections:

| Section | Source | Content |
|---|---|---|
| Base prompt | Static | Core identity, research workflow, response format |
| Tool prompts | Dynamic (per-tool) | Each tool's `prompt()` contribution |
| Citation guidelines | Static | Citation format, diversity, credibility hierarchy |
| Research methodology | Static | Search strategies, when to use research plans |
| Memory section | Dynamic (per-query) | Retrieved memories from previous sessions |
| Date context | Dynamic (per-query) | Current date for recency-aware research |

The prompt is built once per query loop invocation. This is a simpler approach compared to Claude Code's 15+ named sections with memoization and per-turn dynamic rebuilding, but sufficient for a research agent where the prompt doesn't change mid-session.

### 7.2 Tool Prompt Auto-Registration

Each tool contributes its own prompt section via the `prompt()` method. When a new tool is registered, its prompt is automatically included in the system prompt — no manual changes to the context builder required:

```python
def _tool_prompts(self, tools: list[Tool]) -> str:
    tool_sections = []
    for tool in tools:
        prompt = tool.prompt()
        if prompt:
            tool_sections.append(f"### {tool.name}\n{prompt}")
    return "## Available Tools\n\n" + "\n\n".join(tool_sections)
```


## 8. LLM Client

### 8.1 Dual-Model Architecture

The system uses two model tiers (`src/llm/client.py`):

1. **Main model** (`default_model`): Used for the primary research loop — reasoning, tool selection, answer synthesis. Typically a high-capability model (e.g., `claude-opus-4.6`).

2. **Side-query model** (`side_query_model`): Used for cheap, fast auxiliary tasks — content extraction, memory selection, quality classification, context compaction. Typically a smaller model (e.g., `claude-sonnet-4.6`).

Both models are configurable via `settings.yaml`, support custom base URLs (for proxies like vLLM, Ollama, or LiteLLM), and can be pointed at separate endpoints.

### 8.2 Retry and Fallback

The LLM client implements exponential backoff retry with automatic model fallback:

```
Attempt 1: default_model → success? done
         ↓ (retryable error: 429, 5xx, timeout, connection)
Attempt 2: default_model (wait 0.5s) → success? done
         ↓
Attempt 3: default_model (wait 1.0s) → success? done
         ↓
Attempt 4: default_model (wait 2.0s) → success? done
         ↓ (retries exhausted)
Fallback:  fallback_model → success? done
         ↓ (also failed)
Error event emitted
```

**Retryable errors** (mirroring Claude Code's `shouldRetry()`): rate limits (429), overloaded (529), server errors (5xx), connection errors, timeouts.

**Non-retryable errors**: Bad Request (400), Authentication (401/403), Not Found (404), unsupported parameters. These fail immediately without retry.

### 8.3 Side Query Function

The `side_query()` function provides a non-streaming, single-call interface for auxiliary LLM tasks:

```python
async def side_query(
    prompt: str,
    system: str = "",
    model: str | None = None,    # Defaults to side_query_model
    max_tokens: int = 512,
    output_schema: dict | None = None,  # Structured JSON output
) -> str:
```

It supports structured JSON output via `response_format`, which is used by:
- Memory retrieval (select relevant memories)
- Non-research query classification
- Content extraction

A module-level `_shared_config` allows `side_query()` to be called from anywhere (compact.py, retrieval.py, content_extractor.py) without passing the config explicitly.


## 9. Security Considerations

### 9.1 SSRF Prevention

The `web_fetch` tool validates URLs against SSRF (Server-Side Request Forgery) attacks before fetching (`src/utils/url_validator.py`). Blocked targets include:
- Private IP ranges (10.x, 172.16-31.x, 192.168.x)
- Loopback addresses (127.x, ::1)
- Cloud metadata endpoints (169.254.169.254)
- Link-local addresses

The validator resolves DNS to catch hostname-based bypasses (e.g., a domain resolving to 127.0.0.1).

### 9.2 Path Traversal Protection

The `deep_read` tool verifies that the requested `cached_path` is within the configured cache directory before reading. This prevents the LLM from being manipulated into reading arbitrary files:

```python
def _is_path_within_cache(self, cached_path: str, cache_dir: Path) -> bool:
    resolved = Path(cached_path).resolve()
    cache_resolved = cache_dir.resolve()
    return str(resolved).startswith(str(cache_resolved) + os.sep)
```

### 9.3 Rate Limiting

Per-domain rate limiting (`src/utils/rate_limiter.py`) prevents the agent from overwhelming target websites. The default limit is 50 requests per domain per minute, configurable in `settings.yaml`.

### 9.4 API Authentication

The web server supports API key authentication for non-localhost deployments, configurable via `settings.yaml` or the `SEARCH_AGENT_API_KEY` environment variable. Requests must include the key via `Authorization: Bearer <key>` header or `?api_key=<key>` query parameter.


## 10. Configuration

All system parameters are centralized in `config/settings.yaml`:

```yaml
llm:
  default_model: "anthropic/claude-opus-4.6"
  side_query_model: "anthropic/claude-sonnet-4.6"
  fallback_model: "anthropic/claude-sonnet-4.6"
  max_tokens: 128000
  base_url: ""                    # Custom API endpoint
  max_retries: 3
  retry_base_delay_ms: 500

limits:
  max_turns: 50                   # Maximum agentic loop iterations
  compact_threshold_tokens: 80000 # Trigger compaction threshold
  rate_limit_per_domain: 50       # Requests/domain/minute

tools:
  cache_dir: "./cache"
  web_search_default_results: 10
  max_result_size_chars: 20000
  http_timeout: 30
  jina_timeout: 60
  content_extraction_threshold: 15000  # Pre-extraction trigger

hooks:
  min_citations: 2
  min_domains: 2
  min_answer_chars: 200

memory:
  enabled: true
  base_dir: "./memory"
  max_relevant_memories: 5
```


## 11. Data Flow: End-to-End Example

To illustrate the complete system, consider a user query: *"What are the current leading approaches to protein structure prediction, and how do they compare?"*

### Step 1: Session Initialization
- Router receives WebSocket connection
- Memory retrieval loads relevant past memories (e.g., "User has background in computational biology")
- ContextBuilder assembles system prompt with tool prompts, memories, and date context
- `query_loop()` generator is created

### Step 2: Plan Creation (Turn 1)
- LLM receives the query and recognizes it as multi-faceted
- LLM calls `research_plan(action="create")` with sub-tasks:
  1. "Survey current protein structure prediction methods"
  2. "Deep dive into AlphaFold and its successors"
  3. "Investigate competing approaches (ESMFold, RoseTTAFold, etc.)"
  4. "Compare accuracy, speed, and accessibility"
  5. "Synthesize findings and identify trends"
- Loop emits `PLAN_UPDATE` event → frontend renders task tracker

### Step 3: Research Iterations (Turns 2-10)
For each sub-task, the agent:
- Calls `web_search` with targeted queries
- Calls `web_fetch` on promising results → content extraction compresses 50K+ pages to ~4K of key facts
- Calls `cite_source` to register citations
- Calls `research_plan(action="update")` to record findings

### Step 4: Context Management
- After fetching 5+ pages, context approaches 80K tokens
- `should_compact()` triggers microcompact → older tool results truncated to 300 chars
- Pre-extraction has already prevented most context bloat — compaction may not be needed

### Step 5: Answer Synthesis (Turn 11)
- All sub-tasks are `completed` in the research plan
- LLM generates a comprehensive response without tool calls
- Stop hooks run:
  - `PlanCompletenessHook`: ✅ All tasks complete
  - `CitationQualityHook`: ✅ 6 citations registered
  - `SourceDiversityHook`: ✅ Citations from 4 domains
  - `AnswerCompletenessHook`: ✅ Answer is 2,500 chars
- All hooks pass → loop finalizes

### Step 6: Post-Session
- `DONE` event emitted with citations, turn count, session summary
- Memory extraction (fire-and-forget) saves:
  - `[reference]` "Key protein prediction benchmarks: CASP15, CAMEO"
  - `[source_reputation]` "Nature and Science journals provided authoritative reviews"
- Session transcript persisted for continuity


## 12. Related Work and Design Comparisons

### 12.1 Comparison with Claude Code

| Aspect | Claude Code | Our Search Agent |
|---|---|---|
| **Architecture** | Single agentic loop | Single agentic loop (same) |
| **Content handling** | Raw results → post-hoc compaction | Pre-extraction compression → compaction as safety net |
| **System prompt** | 15+ dynamic sections, memoized, per-turn rebuild | 6 static sections, built once per query |
| **Compaction** | Time-based microcompact + cached MC + full compact | Size-based microcompact + full compact |
| **Tool results** | ContentReplacementState (frozen decisions, prompt cache stability) | Per-tool max_result_size_chars + extraction |
| **Research planning** | TodoWrite (generic task tracking) | research_plan (research-specific with findings) |
| **Memory** | CLAUDE.md + memdir/ files | MEMORY.md index + typed memory files |
| **Quality gates** | Hook system (stop hooks) | Hook system (same pattern, research-specific checks) |
| **Interactive tools** | Future-based (AnswerFuture) | AsyncGenerator asend() (simpler, no deadlock risk) |

### 12.2 Key Architectural Decisions

**Why pre-extraction over post-compaction?** Claude Code operates on local files where raw content is small (code files, configs). A web research agent processes web pages that can be 100K+ characters each. Adding 5 raw pages would consume 500K characters — exceeding the context window before any compaction can run. Pre-extraction prevents the problem rather than recovering from it.

**Why a single loop over multi-agent?** Multi-agent architectures (orchestrator-worker, specialized sub-agents) add coordination complexity without proportional benefit for sequential research workflows. The agent's reasoning is inherently serial — each search informs the next. Parallel page fetching is handled within the single loop via `is_concurrency_safe` tool marking.

**Why LLM-based memory retrieval over embedding search?** The memory count is small (dozens, not thousands), making embedding indices unnecessary. LLM-based selection handles semantic nuance (e.g., recognizing that a "source_reputation" memory about arxiv.org is relevant to a machine learning query) better than cosine similarity on embeddings.


## 13. Conclusion

The search agent demonstrates that a single agentic loop, augmented with domain-specific mechanisms for content compression, structured planning, and quality enforcement, can produce thorough, well-cited research responses. The pre-extraction content compression strategy — filtering facts, URLs, and quotes through a cheap LLM before injecting into the main context — enables the agent to process 5-10x more sources per session compared to raw content injection, directly improving answer comprehensiveness and citation diversity.

The system's modular design — with tools, hooks, and memory components as pluggable units — enables straightforward extension: new search backends (e.g., patent search, code search) can be added as tool classes, new quality requirements as hook classes, and new memory types as enum variants, each requiring no changes to the core loop logic.
