"""Opt-in end-to-end research test (needs SearXNG and Ollama, see run.sh)."""

import asyncio
import re

import pytest

from open_deep_research.deep_researcher import deep_researcher

pytestmark = pytest.mark.live

# (question, facts every good report must contain; facts are lowercase substrings)
CASES = {
    "rust": ("What is the Rust programming language, who created it, and what is it mainly used for?", ["graydon hoare", "mozilla"]),
    "webb": ("What is the James Webb Space Telescope, when was it launched, and who built it?", ["2021", "northrop grumman"]),
    "crispr": ("What is CRISPR-Cas9 and who received the 2020 Nobel Prize in Chemistry for it?", ["charpentier", "doudna"]),
    "hungarian": ("Mikor alapították a Budapesti Műszaki és Gazdaságtudományi Egyetemet, és mi az előzménye?", ["1782"]),
}


@pytest.fixture(scope="module", params=CASES)
def result(request):
    question, facts = CASES[request.param]
    config = {"configurable": {"allow_clarification": False, "max_researcher_iterations": 2}}
    output = asyncio.run(deep_researcher.ainvoke({"messages": [{"role": "user", "content": question}]}, config))
    if "NO SOURCES FOUND" in "\n".join(output["raw_notes"]):
        pytest.skip("Search engines returned nothing (rate-limited or blocked); not a quality result")
    return {**output, "facts": facts}


def test_facts_are_retrieved(result):
    retrieved = "\n".join(result["raw_notes"]).lower()
    missing = [f for f in result["facts"] if f not in retrieved]
    assert not missing, f"Search/fetch never retrieved: {missing}"


def test_facts_are_in_report(result):
    report = result["final_report"].lower()
    missing = [f for f in result["facts"] if f not in report]
    assert len(report) > 500 and not missing, f"Report lacks facts: {missing}"


def test_citations_are_grounded(result):
    body, _, sources = result["final_report"].partition("### Sources")
    listed = [int(n) for n in re.findall(r"^\[(\d+)\]", sources, re.M)]
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", body)}
    assert cited and listed == list(range(1, len(listed) + 1)) and cited == set(listed)
    retrieved = "\n".join(result["raw_notes"])
    assert all(url in retrieved for url in re.findall(r"https?://[^\s)]+", sources))
    assert not re.search(r"\[S[0-9a-f]{6}\]", result["final_report"]), "Unresolved source ID"


def unsupported_claims(report: str, raw_notes: list[str]) -> list[str]:
    """Cited sentences whose numbers or proper names appear in none of their cited sources."""
    body, _, sources = report.partition("### Sources")
    urls = {int(n): u for n, u in re.findall(r"^\[(\d+)\].*?(https?://[^\s)]+)", sources, re.M)}
    texts = {
        url: content.lower()
        for url, content in re.findall(r"URL: (\S+)\n(?:[^\n]*\n){3,4}CONTENT:\n(.*?)(?=\n--- SOURCE|\Z)", "\n".join(raw_notes), re.S)
    }
    flagged = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", body):
        if sentence.lstrip().startswith("#"):
            continue
        cited = [urls[int(n)] for n in re.findall(r"\[(\d+)\]", sentence) if int(n) in urls]
        tokens = re.findall(r"\b\d+(?:[.,]\d+)*\b|\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", re.sub(r"\[\d+\]", "", sentence))
        if cited and any(not any(t.lower() in texts.get(u, "") for u in cited) for t in tokens):
            flagged.append(sentence)
    return flagged


def test_claims_are_supported_by_cited_sources(result):
    flagged = unsupported_claims(result["final_report"], result["raw_notes"])
    total = len(re.findall(r"\[\d+\]", result["final_report"].partition("### Sources")[0]))
    assert len(flagged) <= max(2, total // 5), f"Unsupported claims: {flagged}"
