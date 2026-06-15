# SearchClaw

SearchClaw is an agentic web research tool that searches, reads, and synthesizes answers with source-aware tooling.

## Install

Use Python 3.11 or newer.

```bash
pip install -e .
```

Set at least one LLM provider key:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# or
export OPENAI_API_KEY="sk-..."
```

Recommended search/fetch keys:

```bash
export SERPER_API_KEY="..."
export JINA_API_KEY="..."
```

## Run

```bash
python -m src.main
```

By default the server binds to `127.0.0.1:8000`. Open `http://localhost:8000`.

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
