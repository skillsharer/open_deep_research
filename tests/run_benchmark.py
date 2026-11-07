"""Run a fixed subset of DeepResearch Bench prompts with the local system."""

import argparse
import asyncio
import json
import logging
import re
import uuid
from pathlib import Path
from open_deep_research.deep_researcher import deep_researcher


async def run(prompt: str) -> dict:
    run_id = uuid.uuid4().hex[:12]
    config = {"configurable": {"allow_clarification": False, "max_researcher_iterations": 3}}
    result = await deep_researcher.ainvoke(
        {"messages": [{"role": "user", "content": prompt}], "run_id": run_id}, config
    )
    raw_notes = result.get("raw_notes", [])
    source_count = len(set(re.findall(r"--- SOURCE \[S[0-9a-f]{6}\]:", "\n".join(raw_notes))))
    return {
        "report": result["final_report"],
        "research_plan": result.get("research_plan", []),
        "research_errors": result.get("research_errors", []),
        "raw_notes_count": len(raw_notes),
        "source_count": source_count,
        "run_id": run_id,
        "evidence_block_count": len(result.get("evidence_blocks", [])),
        "report_dimension_ids": result.get("report_dimension_ids", []),
        "report_evidence_markers": result.get("report_evidence_markers", []),
    }


def main() -> None:
    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("open_deep_research.deep_researcher").setLevel(logging.INFO)
    logging.getLogger(__name__).setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="tests/expt_results/deep_research_bench_gpt-5.jsonl")
    parser.add_argument("--output", default="tests/expt_results/local_benchmark.jsonl")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    reference = [json.loads(line) for line in Path(args.reference).read_text().splitlines()]
    output = Path(args.output)
    with output.open("w", encoding="utf-8") as file:
        for case in reference[: args.limit]:
            logging.getLogger(__name__).info("Starting benchmark case %s", case["id"])
            run_result = asyncio.run(run(case["prompt"]))
            file.write(json.dumps({
                "id": case["id"],
                "prompt": case["prompt"],
                **run_result,
            }) + "\n")


if __name__ == "__main__":
    main()
