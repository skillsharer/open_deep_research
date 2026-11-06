from open_deep_research.deep_researcher import (
    _covered_requirement_numbers,
    _missing_report_requirements,
)
from open_deep_research.state import CreateResearchPlan


def test_requirement_evidence_requires_every_numbered_requirement():
    notes = (
        "Requirement 1: creator\n"
        "Evidence: Graydon Hoare\n"
        "Sources: [Sabc123]\n"
        "Requirement 2: sponsor\n"
        "Evidence: Mozilla\n"
        "Sources: [Sdef456]"
    )

    assert _covered_requirement_numbers(
        ["creator", "sponsor"], notes, ["Sabc123", "Sdef456"]
    ) == {1, 2}
    assert _covered_requirement_numbers(
        ["creator", "sponsor"], notes.split("Requirement 2:")[0], ["Sabc123"]
    ) == {1}


def test_empty_evidence_requirements_are_not_covered():
    assert _covered_requirement_numbers(
        ["creator"], "Requirement 1: creator\nEvidence: x\nSources: [Sabc123]", []
    ) == set()


def test_research_plan_requires_atomic_requirements():
    plan = CreateResearchPlan(dimensions=[{
        "dimension_id": "dimension-001",
        "subject": "Rust",
        "aspect": "origins",
        "deliverable": "identify the creator",
        "research_topic": "Rust origins",
        "evidence_requirements": ["identify the creator"],
    }])

    assert plan.dimensions[0].evidence_requirements == ["identify the creator"]


def test_report_coverage_detects_missing_markers():
    plan = [{
        "id": "dimension-001",
        "covered_requirements": [1, 2],
    }]

    assert _missing_report_requirements(
        "<!-- evidence: dimension-001:1 -->", plan
    ) == ["dimension-001:2"]
