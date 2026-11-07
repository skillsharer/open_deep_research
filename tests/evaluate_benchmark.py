"""Deterministically score local reports against fixed benchmark dimensions."""

import argparse
import json
import re
import sys
from pathlib import Path

STOPWORDS = {"about", "after", "analysis", "and", "from", "into", "that", "the", "their", "what", "with"}


def dimensions(article: str) -> list[str]:
    return [heading[3:].strip() for heading in re.findall(r"^## .+$", article, re.M)]


def score(report: str, rubric: list[str]) -> dict[str, int]:
    text = report.lower()
    covered = 0
    for item in rubric:
        terms = {term for term in re.findall(r"[a-z0-9]{4,}", item.lower()) if term not in STOPWORDS}
        covered += bool(terms and sum(term in text for term in terms) / len(terms) >= 0.5)
    body, _, sources = report.partition("### Sources")
    citations = bool(re.findall(r"\[\d+\]", body)) and bool(re.findall(r"^\[\d+\].*https?://", sources, re.M))
    return {
        "dimensions": len(rubric),
        "covered": covered,
        "citation_structure": int(citations),
        "length": len(report),
    }


def diagnostics(result: dict) -> str:
    plan = result.get("research_plan", [])
    unique_ids = {item.get("id") for item in plan}
    terminal = sum(item.get("status") in {"covered", "unavailable"} for item in plan)
    errors = len(result.get("research_errors", []))
    expected_ids = {item.get("id") for item in plan}
    reported_ids = set(result.get("report_dimension_ids", []))
    marker_coverage = len(expected_ids & reported_ids)
    return (
        f"planned={len(unique_ids)}/{len(plan)}, "
        f"terminal={terminal}, markers={marker_coverage}/{len(expected_ids)}, "
        f"evidence_blocks={result.get('evidence_block_count', 0)}, "
        f"sources={result.get('source_count', 0)}, errors={errors}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="tests/expt_results/deep_research_bench_gpt-5.jsonl")
    parser.add_argument("--reports", default="tests/expt_results/local_benchmark.jsonl")
    args = parser.parse_args()

    references = {str(item["id"]): item for item in map(json.loads, Path(args.reference).read_text().splitlines())}
    for line in Path(args.reports).read_text().splitlines():
        report = json.loads(line)
        rubric = dimensions(references[str(report["id"])]["article"])
        result = score(report["report"], rubric)
        baseline = score(references[str(report["id"])]["article"], rubric)
        sys.stdout.write(
            f'{report["id"]}: local={result["covered"]}/{result["dimensions"]}, '
            f'reference={baseline["covered"]}/{baseline["dimensions"]}, '
            f'citations={result["citation_structure"]}, length={result["length"]}, '
            f'{diagnostics(report)}\n'
        )


if __name__ == "__main__":
    main()
