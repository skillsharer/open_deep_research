import asyncio
import logging

import pytest
from langchain_core.messages import AIMessage

import open_deep_research.deep_researcher as researcher_module

from open_deep_research.deep_researcher import (
    _covered_requirement_numbers,
    _findings_for_report,
    _has_source_backed_citations,
    _missing_report_dimensions,
    _missing_report_requirements,
    _merge_report_repair,
    _fallback_research_plan,
    _plan_items_from_dimensions,
    _pending_dimensions,
    _source_ids,
)
from open_deep_research.state import (
    CreateResearchPlan,
    ResearchPlanDimension,
    override_reducer,
)


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
        "search_queries": ["Rust creator"],
    }])

    assert plan.dimensions[0].evidence_requirements == ["identify the creator"]


def test_plan_items_preserve_required_structured_fields():
    dimension = ResearchPlanDimension(
        dimension_id="D1",
        subject="Rust",
        aspect="origins",
        deliverable="identify the creator",
        research_topic="Rust origins",
        evidence_requirements=["identify the creator"],
        search_queries=["Rust creator"],
    )

    item = _plan_items_from_dimensions([dimension])[0]

    assert item["id"] == "D1"
    assert item["evidence_requirements"] == ["identify the creator"]
    assert item["queries"] == ["Rust creator"]


def test_fallback_plan_is_researchable():
    plan = _fallback_research_plan({"research_brief": "Rust origins"})

    assert plan[0]["status"] == "pending"
    assert plan[0]["evidence_requirements"] == ["Rust origins"]
    assert plan[0]["queries"] == ["Rust origins"]


def test_planning_logs_progress(monkeypatch, caplog):
    dimension = ResearchPlanDimension(
        dimension_id="D1",
        subject="Rust",
        aspect="origins",
        deliverable="identify the creator",
        research_topic="Rust origins",
        evidence_requirements=["identify the creator"],
        search_queries=["Rust creator"],
    )

    class Planner:
        def with_structured_output(self, schema):
            return self

        def with_retry(self, **kwargs):
            return self

        def with_config(self, config):
            return self

        async def ainvoke(self, messages):
            return CreateResearchPlan(dimensions=[dimension])

    monkeypatch.setattr(researcher_module, "configurable_model", Planner())
    with caplog.at_level(logging.INFO, logger=researcher_module.__name__):
        asyncio.run(researcher_module.supervisor({"supervisor_messages": []}, {}))

    assert "Creating research plan" in caplog.text
    assert "Research plan created: 1 dimensions" in caplog.text


def test_pending_report_logs_incomplete_state(caplog):
    state = {"research_plan": [{"id": "D1", "status": "pending"}]}
    with caplog.at_level(logging.WARNING, logger=researcher_module.__name__):
        result = asyncio.run(researcher_module.final_report_generation(state, {}))

    assert "pending dimensions were not resolved: D1" in caplog.text
    assert result["final_report"].startswith("Research incomplete")


def test_dimension_attempt_and_result_are_logged(monkeypatch, caplog):
    class Researcher:
        async def ainvoke(self, state, config):
            return {
                "research_dimension_id": "D1",
                "raw_notes": ["--- SOURCE [Sabc123]: source ---"],
                "evidence_blocks": ["Requirement 1: creator\nEvidence: Hoare\nSources: [Sabc123]"],
                "compressed_research": "Findings",
            }

    dimension = ResearchPlanDimension(
        dimension_id="D1", subject="Rust", aspect="origins",
        deliverable="creator", research_topic="Rust origins",
        evidence_requirements=["creator"], search_queries=["Rust creator"],
    )
    state = {
        "supervisor_messages": [AIMessage(content="")],
        "research_plan": _plan_items_from_dimensions([dimension]),
        "research_iterations": 1,
    }
    monkeypatch.setattr(researcher_module, "researcher_subgraph", Researcher())
    with caplog.at_level(logging.INFO, logger=researcher_module.__name__):
        asyncio.run(researcher_module.supervisor_tools(state, {}))

    assert "Researching dimension D1 (attempt 1/3)" in caplog.text
    assert "Dimension D1: covered, requirements 1/1, sources 1" in caplog.text


def test_report_coverage_detects_missing_markers():
    plan = [{
        "id": "dimension-001",
        "covered_requirements": [1, 2],
    }]

    assert _missing_report_requirements(
        "<!-- evidence: dimension-001:1 -->", plan
    ) == ["dimension-001:2"]


def test_research_plan_override_replaces_previous_state():
    previous = [{"id": "dimension-001"}]
    current = [{"id": "dimension-002"}]

    assert override_reducer(previous, {"type": "override", "value": current}) == current


def test_research_plan_rejects_duplicate_dimensions():
    dimension = {
        "dimension_id": "dimension-001",
        "subject": "Rust",
        "aspect": "origins",
        "deliverable": "identify the creator",
        "research_topic": "Rust origins",
        "evidence_requirements": ["identify the creator"],
        "search_queries": ["Rust creator"],
    }

    with pytest.raises(ValueError):
        CreateResearchPlan(dimensions=[dimension, {**dimension, "dimension_id": "dimension-002"}])


def test_research_plan_requires_targeted_queries():
    with pytest.raises(ValueError):
        CreateResearchPlan(dimensions=[{
            "dimension_id": "dimension-001",
            "subject": "Rust",
            "aspect": "origins",
            "deliverable": "identify the creator",
            "research_topic": "Rust origins",
            "evidence_requirements": ["identify the creator"],
        }])


def test_report_coverage_detects_missing_dimensions():
    plan = [{"id": "dimension-001"}, {"id": "dimension-002"}]

    assert _missing_report_dimensions(
        "<!-- dimension: dimension-001 -->", plan
    ) == ["dimension-002"]


def test_report_repair_preserves_existing_dimensions():
    plan = [{"id": "D1"}, {"id": "D2"}]
    original = "<!-- dimension: D1 --> Existing"
    repaired = "<!-- dimension: D2 --> Added"

    merged = _merge_report_repair(original, repaired, plan)

    assert "D1" in merged
    assert "D2" in merged


def test_pending_dimensions_include_in_progress_work():
    plan = [
        {"id": "dimension-001", "status": "in_progress"},
        {"id": "dimension-002", "status": "covered"},
    ]

    assert [item["id"] for item in _pending_dimensions(plan)] == ["dimension-001"]


def test_source_ids_require_structured_source_records():
    notes = "--- SOURCE [Sabc123]: source ---\n--- SOURCE [Sdef456]: source ---"

    assert _source_ids([notes]) == {"Sabc123", "Sdef456"}
    assert _source_ids(["search returned no results"]) == set()


def test_raw_evidence_is_preserved_when_compressed_notes_are_empty():
    raw_notes = "--- SOURCE [Sabc123]: source ---\nCONTENT:\nEvidence"

    assert _findings_for_report([], [], [raw_notes]) == raw_notes
    assert _findings_for_report([], ["Structured evidence"], [raw_notes]) == "Structured evidence"
    assert _findings_for_report(["Compressed evidence"], ["Structured evidence"], [raw_notes]) == "Compressed evidence"


def test_source_backed_report_requires_citations():
    raw_notes = "--- SOURCE [Sabc123]: source ---\nURL: https://example.com"
    cited_report = "Finding [1]\n\n### Sources\n[1] source: https://example.com"
    uncited_report = "Finding without a citation"

    assert _has_source_backed_citations(cited_report, [raw_notes])
    assert not _has_source_backed_citations(uncited_report, [raw_notes])
    assert _has_source_backed_citations(uncited_report, ["NO SOURCES FOUND"])
