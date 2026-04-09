"""
Benchmark script for the ReAct baseline agent.

Reads problems from a JSONL file, runs each through the ReAct agent,
and records the results.

Usage:
    python baseline/react/run_benchmark.py                         # Problems 1-10
    python baseline/react/run_benchmark.py --start 11 --end 20     # Problems 11-20
    python baseline/react/run_benchmark.py --file decrypted_problems_zh.jsonl
    python baseline/react/run_benchmark.py --max-search 10 --max-fetch 15

No server required -- runs the agent directly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path so we can import config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from baseline.react.agent import react_agent


def load_problems(path: str, start: int, end: int) -> list[dict]:
    """Load problems from line `start` to `end` (1-indexed, inclusive)."""
    problems = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if i < start:
                continue
            if i > end:
                break
            line = line.strip()
            if not line:
                continue
            problems.append(json.loads(line))
    return problems


async def run(args):
    # Resolve problem file path
    tests_dir = Path(__file__).parent.parent.parent / "tests"
    problem_path = tests_dir / args.file
    if not problem_path.exists():
        print(f"Error: Problem file not found: {problem_path}")
        sys.exit(1)

    problems = load_problems(str(problem_path), args.start, args.end)
    print(f"Loaded {len(problems)} problems from {args.file} (lines {args.start}-{args.end})")
    print(f"Max turns: {args.max_turns}, max search: {args.max_search}, max fetch: {args.max_fetch}")
    print(f"Output: {args.output}")
    print(f"{'='*70}\n")

    results = []
    total_time = 0
    output_path = Path(args.output)

    for i, problem in enumerate(problems):
        problem_num = args.start + i
        query = problem["problem"]
        ground_truth = problem["answer"]
        topic = problem.get("problem_topic", "Unknown")

        print(f"[#{problem_num}, {i+1}/{len(problems)}] Topic: {topic}")
        print(f"  Q: {query[:120]}{'...' if len(query) > 120 else ''}")
        print(f"  Expected: {ground_truth}")

        start = time.time()
        try:
            response = await react_agent(
                query=query,
                max_turns=args.max_turns,
                max_search=args.max_search,
                max_fetch=args.max_fetch,
            )
            elapsed = time.time() - start
            total_time += elapsed

            answer = response.get("answer", "")
            turn_count = response.get("turn_count", 0)
            search_count = response.get("search_count", 0)
            fetch_count = response.get("fetch_count", 0)

            print(f"  Answer: {answer[:200]}{'...' if len(answer) > 200 else ''}")
            print(f"  turns={turn_count}, searches={search_count}, fetches={fetch_count}, time={elapsed:.1f}s")

            results.append({
                "index": problem_num,
                "topic": topic,
                "query": query,
                "ground_truth": ground_truth,
                "predicted": answer,
                "turn_count": turn_count,
                "search_count": search_count,
                "fetch_count": fetch_count,
                "elapsed_seconds": round(elapsed, 1),
            })

        except Exception as e:
            elapsed = time.time() - start
            total_time += elapsed
            print(f"  ERROR: {e} (time={elapsed:.1f}s)")
            results.append({
                "index": problem_num,
                "topic": topic,
                "query": query,
                "ground_truth": ground_truth,
                "predicted": "",
                "error": str(e),
                "elapsed_seconds": round(elapsed, 1),
            })

        # Incremental save
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(results[-1], ensure_ascii=False) + "\n")

        print()

    # Summary
    print(f"{'='*70}")
    print(f"Completed {len(problems)} problems")
    print(f"Total time: {total_time:.1f}s, avg: {total_time/len(problems):.1f}s per query")
    print(f"Results saved to {output_path}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="ReAct baseline benchmark")
    parser.add_argument("--file", default="decrypted_problems.jsonl",
                        help="Problem file name in tests/ directory")
    parser.add_argument("--start", type=int, default=1,
                        help="Start line number, 1-indexed inclusive (default: 1)")
    parser.add_argument("--end", type=int, default=10,
                        help="End line number, 1-indexed inclusive (default: 10)")
    parser.add_argument("--max-turns", type=int, default=120,
                        help="Maximum agent loop turns (default: 120)")
    parser.add_argument("--max-search", type=int, default=50,
                        help="Maximum search calls (default: 50)")
    parser.add_argument("--max-fetch", type=int, default=50,
                        help="Maximum fetch calls (default: 50)")
    parser.add_argument("--output", default="baseline/react/results.jsonl",
                        help="Output file (default: baseline/react/results.jsonl)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
