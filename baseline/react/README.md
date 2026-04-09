# ReAct Baseline Agent

A minimal ReAct (Reasoning + Acting) agent for benchmark comparison against the full SearchClaw system.

## What This Is

A plain ReAct loop with **no harness engineering**:
- No quality hooks or answer validation
- No research plan generation
- No content extraction or compression
- No memory or context compaction
- No citation management
- No user interaction (ask_user)

Just: **Think -> Tool -> Observe -> Repeat**.

## Files

| File | Description |
|------|-------------|
| `agent.py` | ReAct agent loop with `web_search` and `web_fetch` tools |
| `run_benchmark.py` | Benchmark script that reads JSONL problems and records results |

## Usage

### Run benchmark (problems 1-10)

```bash
python baseline/react/run_benchmark.py
```

### Custom range

```bash
python baseline/react/run_benchmark.py --start 11 --end 20
```

### All options

```bash
python baseline/react/run_benchmark.py \
    --file decrypted_problems.jsonl \
    --start 1 \
    --end 50 \
    --max-turns 100 \
    --max-search 30 \
    --max-fetch 30 \
    --output baseline/react/results.jsonl
```

| Flag | Default | Description |
|------|---------|-------------|
| `--file` | `decrypted_problems.jsonl` | Problem file in `tests/` directory |
| `--start` | `1` | Start line number (1-indexed, inclusive) |
| `--end` | `10` | End line number (1-indexed, inclusive) |
| `--max-turns` | `100` | Maximum agent loop turns |
| `--max-search` | `30` | Maximum web search calls |
| `--max-fetch` | `30` | Maximum web fetch calls |
| `--output` | `baseline/react/results.jsonl` | Output file path |

## Configuration

Uses the same `config/settings.yaml` as the main system for LLM model and API settings. No additional configuration needed.

## Output Format

Results are saved as JSONL, one line per problem:

```json
{
    "index": 1,
    "topic": "Science",
    "query": "What is ...?",
    "ground_truth": "expected answer",
    "predicted": "agent's answer",
    "turn_count": 12,
    "search_count": 5,
    "fetch_count": 3,
    "elapsed_seconds": 45.2
}
```
