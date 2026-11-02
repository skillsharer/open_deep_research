"""Run a fixed subset of DeepResearch Bench prompts with the local system."""

import argparse
import asyncio
import json
from pathlib import Path

from open_deep_research.deep_researcher import deep_researcher


async def run(prompt: str) -> str:
    config = {"configurable": {"allow_clarification": False, "max_researcher_iterations": 3}}
    result = await deep_researcher.ainvoke(
        {"messages": [{"role": "user", "content": prompt}]}, config
    )
    return result["final_report"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="tests/expt_results/deep_research_bench_gpt-5.jsonl")
    parser.add_argument("--output", default="tests/expt_results/local_benchmark.jsonl")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    reference = [json.loads(line) for line in Path(args.reference).read_text().splitlines()]
    output = Path(args.output)
    with output.open("w", encoding="utf-8") as file:
        for case in reference[: args.limit]:
            report = asyncio.run(run(case["prompt"]))
            file.write(json.dumps({"id": case["id"], "prompt": case["prompt"], "report": report}) + "\n")


if __name__ == "__main__":
    main()
