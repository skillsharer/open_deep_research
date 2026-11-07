"""Main LangGraph implementation for the Deep Research agent."""

import asyncio
import logging
import re
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from open_deep_research.configuration import (
    Configuration,
)
from open_deep_research.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_plan_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
)
from open_deep_research.state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    CreateResearchPlan,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
    SupervisorState,
)
from open_deep_research.utils import (
    get_all_tools,
    get_model_token_limit,
    finalize_citations,
    get_notes_from_tool_calls,
    get_today_str,
    is_token_limit_exceeded,
    remove_up_to_last_ai_message,
    think_tool,
)

# Initialize a configurable model that we will use throughout the agent
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "num_predict"),
)
logger = logging.getLogger(__name__)
MAX_DIMENSION_ATTEMPTS = 3

def _covered_requirement_numbers(
    requirements: list[str],
    raw_notes: str,
    source_ids: list[str],
) -> set[int]:
    """Return numbered requirements with source-backed evidence blocks."""
    if not requirements or not raw_notes or not source_ids:
        return set()
    blocks = re.split(r"(?=Requirement \d+:)", raw_notes)
    covered_numbers = {
        int(match.group(1))
        for block in blocks
        if "Evidence:" in block
        and re.search(r"Sources?:.*S[0-9a-f]{6}", block)
        for match in [re.match(r"Requirement (\d+):", block)]
        if match
    }
    return covered_numbers


def _missing_report_requirements(
    report: str,
    research_plan: list[dict],
) -> list[str]:
    """Find requirement markers omitted from a generated report."""
    markers = set(re.findall(r"<!-- evidence: ([^ ]+:\d+) -->", report))
    return [
        f"{item['id']}:{number}"
        for item in research_plan
        for number in item["covered_requirements"]
        if f"{item['id']}:{number}" not in markers
    ]


def _missing_report_dimensions(
    report: str,
    research_plan: list[dict],
) -> list[str]:
    """Find research-plan dimensions omitted from a generated report."""
    markers = set(re.findall(r"<!-- dimension: ([^ ]+) -->", report))
    return [
        item["id"]
        for item in research_plan
        if item["id"] not in markers
    ]


def _merge_report_repair(
    original: str,
    repaired: str,
    research_plan: list[dict],
) -> str:
    """Preserve the original report unless the repair returns a complete report."""
    expected_dimensions = {item["id"] for item in research_plan}
    repaired_dimensions = set(re.findall(r"<!-- dimension: ([^ ]+) -->", repaired))
    if expected_dimensions.issubset(repaired_dimensions):
        return repaired
    return f"{original}\n\n{repaired}"


def _research_plan_errors(plan: list[dict]) -> list[str]:
    """Validate persisted plan invariants before execution or report generation."""
    errors = []
    ids = [item.get("id") for item in plan]
    subjects = [
        (item.get("subject", "").casefold(), item.get("aspect", "").casefold())
        for item in plan
    ]
    if len(ids) != len(set(ids)):
        errors.append("research plan contains duplicate dimension IDs")
    if len(subjects) != len(set(subjects)):
        errors.append("research plan contains duplicate subject/aspect pairs")
    for item in plan:
        if not all(item.get(field, "").strip() for field in ("id", "subject", "aspect", "deliverable")):
            errors.append(f"{item.get('id', 'unknown')} has empty required fields")
        if item.get("status") not in {"pending", "in_progress", "covered", "unavailable"}:
            errors.append(f"{item.get('id', 'unknown')} has an invalid status")
        if item.get("attempts", -1) < 0:
            errors.append(f"{item.get('id', 'unknown')} has invalid attempts")
        if not set(item.get("covered_requirements", [])).issubset(
            set(range(1, len(item.get("evidence_requirements", [])) + 1))
        ):
            errors.append(f"{item.get('id', 'unknown')} has invalid covered requirements")
    return errors


def _findings_for_report(
    notes: list[str],
    evidence_blocks: list[str],
    raw_notes: list[str],
) -> str:
    """Prefer compressed findings, but preserve raw evidence after partial failures."""
    return "\n".join(notes or evidence_blocks or raw_notes)


def _has_source_backed_citations(report: str, raw_notes: list[str]) -> bool:
    """Check that retrieved source records are represented in the final report."""
    has_sources = bool(re.search(r"--- SOURCE \[S[0-9a-f]{6}\]:", "\n".join(raw_notes)))
    if not has_sources:
        return True
    body, _, sources = report.partition("### Sources")
    return bool(re.findall(r"\[\d+\]", body)) and bool(
        re.findall(r"^\[\d+\].*https?://", sources, re.M)
    )


def _source_ids(notes: list[str]) -> set[str]:
    return set(re.findall(r"--- SOURCE \[(S[0-9a-f]{6})\]:", "\n".join(notes)))


def _pending_dimensions(plan: list[dict]) -> list[dict]:
    return [
        item for item in plan
        if item.get("status") in {"pending", "in_progress"}
    ]


def _plan_items_from_dimensions(dimensions) -> list[dict]:
    return [
        {
            "id": dimension.dimension_id,
            "subject": dimension.subject,
            "aspect": dimension.aspect,
            "deliverable": dimension.deliverable,
            "requirement": dimension.research_topic,
            "evidence_requirements": dimension.evidence_requirements,
            "queries": dimension.search_queries,
            "status": "pending",
            "source_ids": [],
            "attempts": 0,
            "covered_requirements": [],
        }
        for dimension in dimensions
    ]


def _fallback_research_plan(state: SupervisorState) -> list[dict]:
    brief = state.get("research_brief", "").strip()
    topic = brief or "the user's research request"
    return [{
        "id": "D1",
        "subject": topic[:120],
        "aspect": "answer",
        "deliverable": topic,
        "requirement": topic,
        "evidence_requirements": [topic],
        "queries": [topic],
        "status": "pending",
        "source_ids": [],
        "attempts": 0,
        "covered_requirements": [],
    }]


async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "__end__"]]:
    """Analyze user messages and ask clarifying questions if the research scope is unclear.
    
    This function determines whether the user's request needs clarification before proceeding
    with research. If clarification is disabled or not needed, it proceeds directly to research.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings and preferences
        
    Returns:
        Command to either end with a clarifying question or proceed to research brief
    """
    # Step 1: Check if clarification is enabled in configuration
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        # Skip clarification step and proceed directly to research
        return Command(goto="write_research_brief")
    
    # Step 2: Prepare the model for structured clarification analysis
    messages = state["messages"]
    model_limit_param = "num_predict" if configurable.research_model.startswith("ollama:") else "max_tokens"
    model_config = {
        "model": configurable.research_model,
        model_limit_param: configurable.research_model_max_tokens,
        "tags": ["langsmith:nostream"]
    }
    
    # Configure model with structured output and retry logic
    clarification_model = (
        configurable_model
        .with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )
    
    # Step 3: Analyze whether clarification is needed
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages), 
        date=get_today_str()
    )
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])
    
    # Step 4: Route based on clarification analysis
    if response.need_clarification:
        # End with clarifying question for user
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        # Proceed to research with verification message
        return Command(
            goto="write_research_brief", 
            update={"messages": [AIMessage(content=response.verification)]}
        )


async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """Transform user messages into a structured research brief and initialize supervisor.
    
    This function analyzes the user's messages and generates a focused research brief
    that will guide the research supervisor. It also sets up the initial supervisor
    context with appropriate prompts and instructions.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to research supervisor with initialized context
    """
    logger.info("Preparing research brief")
    # Step 1: Set up the research model for structured output
    configurable = Configuration.from_runnable_config(config)
    researcher_limit_param = "num_predict" if configurable.research_model.startswith("ollama:") else "max_tokens"
    research_model_config = {
        "model": configurable.research_model,
        researcher_limit_param: configurable.research_model_max_tokens,
        "tags": ["langsmith:nostream"]
    }
    
    # Configure model for structured research question generation
    research_model = (
        configurable_model
        .with_structured_output(ResearchQuestion)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 2: Generate structured research brief from user messages
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    )
    response = await research_model.ainvoke([HumanMessage(content=prompt_content)])
    
    # Step 3: Initialize supervisor with research brief and instructions
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations
    )
    
    return Command(
        goto="research_supervisor", 
        update={
            "research_brief": response.research_brief,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=response.research_brief)
                ]
            }
        }
    )


async def supervisor(
    state: SupervisorState,
    config: RunnableConfig,
) -> Command[Literal["supervisor", "supervisor_tools"]]:
    """Lead research supervisor that plans research strategy and delegates to researchers.
    
    The supervisor analyzes the research brief and decides how to break down the research
    into manageable tasks. It can use think_tool for strategic planning, ConductResearch
    to delegate tasks to sub-researchers, or ResearchComplete when satisfied with findings.
    
    Args:
        state: Current supervisor state with messages and research context
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to supervisor_tools for tool execution
    """
    # Step 1: Configure the supervisor model with available tools
    configurable = Configuration.from_runnable_config(config)
    supervisor_limit_param = "num_predict" if configurable.research_model.startswith("ollama:") else "max_tokens"
    research_model_config = {
        "model": configurable.research_model,
        supervisor_limit_param: configurable.research_model_max_tokens,
        "tags": ["langsmith:nostream"]
    }

    if not state.get("research_plan"):
        logger.info("Creating research plan")
        planner_model = (
            configurable_model
            .with_structured_output(CreateResearchPlan)
            .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
            .with_config(research_model_config)
        )
        planner_messages = [
            SystemMessage(content=research_plan_prompt),
            *state.get("supervisor_messages", []),
        ]
        try:
            plan_response = await planner_model.ainvoke(planner_messages)
        except Exception as first_error:
            logger.warning(
                "Research plan generation failed; retrying with schema repair",
                extra={"run_id": state.get("run_id", "unknown")},
                exc_info=True,
            )
            repair_prompt = (
                "The previous research plan was invalid. Generate the complete plan again "
                "using the structured schema. Every dimension MUST include all fields: "
                "dimension_id, subject, aspect, deliverable, research_topic, "
                "evidence_requirements (non-empty list of atomic strings), and "
                "search_queries (non-empty list of targeted strings). Do not omit any "
                "field and do not return prose outside the structured response.\n"
                f"Validation failure: {type(first_error).__name__}: {first_error}"
            )
            try:
                plan_response = await planner_model.ainvoke(
                    planner_messages + [HumanMessage(content=repair_prompt)]
                )
            except Exception:
                logger.exception(
                    "Research plan schema repair failed; using deterministic fallback",
                    extra={"run_id": state.get("run_id", "unknown")},
                )
                research_plan = _fallback_research_plan(state)
                plan_context = (
                    "Planner recovery fallback used after structured-output validation "
                    "failed. Research this dimension and preserve the error in diagnostics:\n"
                    f"- {research_plan[0]['id']}: {research_plan[0]['requirement']}"
                )
                return Command(
                    goto="supervisor",
                    update={
                        "research_plan": research_plan,
                        "research_errors": [
                            "Research plan generation failed validation; deterministic "
                            "fallback plan was used"
                        ],
                        "supervisor_messages": [HumanMessage(content=plan_context)],
                    },
                )
        research_plan = _plan_items_from_dimensions(plan_response.dimensions)
        logger.info("Research plan created: %d dimensions", len(research_plan))
        plan_context = "\n".join(
            f"- {item['id']}: {item['requirement']} "
            f"(requirements: {item['evidence_requirements']})"
            for item in research_plan
        )
        return Command(
            goto="supervisor",
            update={
                "research_plan": research_plan,
                "supervisor_messages": [
                    HumanMessage(content=(
                        "Complete research plan created. Research every pending dimension:\n"
                        f"{plan_context}"
                    ))
                ],
            },
        )
    
    # Available tools: research delegation, completion signaling, and strategic thinking
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]
    
    # Configure model with tools, retry logic, and model settings
    research_model = (
        configurable_model
        .bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 2: Generate supervisor response based on current context
    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(supervisor_messages)
    
    # Step 3: Update state and proceed to tool execution
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )

async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """Execute tools called by the supervisor, including research delegation and strategic thinking.
    
    This function handles three types of supervisor tool calls:
    1. think_tool - Strategic reflection that continues the conversation
    2. ConductResearch - Delegates research tasks to sub-researchers
    3. ResearchComplete - Signals completion of research phase
    
    Args:
        state: Current supervisor state with messages and iteration count
        config: Runtime configuration with research limits and model settings
        
    Returns:
        Command to either continue supervision loop or end research phase
    """
    # Step 1: Extract current state and check exit conditions
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]
    
    # Define exit criteria for research phase
    max_iterations = max(
        configurable.max_researcher_iterations,
        min(len(state.get("research_plan", [])) * 3 + 1, 20),
    )
    exceeded_allowed_iterations = research_iterations > max_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    plan_items = state.get("research_plan", [])
    has_research_results = bool(_source_ids(state.get("raw_notes", [])))
    has_pending_dimensions = bool(_pending_dimensions(plan_items))

    # A model may stop calling tools before the deterministic plan is complete.
    # Convert that idle turn into a bounded follow-up for the next pending item.
    if (
        (no_tool_calls or research_complete_tool_call)
        and has_pending_dimensions
        and not exceeded_allowed_iterations
    ):
        no_tool_calls = False
        research_complete_tool_call = False
        most_recent_message = AIMessage(content="")

    if (research_complete_tool_call or (no_tool_calls and not has_pending_dimensions)) and (
        not has_research_results or has_pending_dimensions
    ):
        if research_iterations >= configurable.max_researcher_iterations:
            for item in plan_items:
                if item["status"] in {"pending", "in_progress"}:
                    item["status"] = "unavailable"
            return Command(
                goto=END,
                update={
                    "notes": get_notes_from_tool_calls(supervisor_messages),
                    "research_brief": state.get("research_brief", ""),
                    "research_plan": {
                        "type": "override",
                        "value": plan_items,
                    },
                    "research_errors": [
                        f"Research dimension {item['id']} was unavailable after "
                        f"{item.get('attempts', 0)} attempt(s)"
                        for item in plan_items
                        if item["status"] == "unavailable"
                        and not item.get("covered_requirements")
                    ],
                }
            )
        return Command(
            goto="supervisor",
            update={
                "supervisor_messages": [
                    ToolMessage(
                        content=(
                            "Research cannot be completed yet: "
                            + (
                                "no researcher returned evidence. "
                                "Delegate at least one focused ConductResearch task."
                                if not has_research_results
                                else "some research-plan dimensions are still pending. "
                                "Delegate focused follow-up tasks for them."
                            )
                        ),
                        name="research_validation",
                        tool_call_id="research_validation"
                    )
                ]
            }
        )

    # Exit if any termination condition is met
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        for item in plan_items:
            if item["status"] in {"pending", "in_progress"}:
                item["status"] = "unavailable"
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", ""),
                "research_plan": {
                    "type": "override",
                    "value": plan_items,
                },
                "research_errors": [
                    f"Research dimension {item['id']} was unavailable after "
                    f"{item.get('attempts', 0)} attempt(s)"
                    for item in plan_items
                    if item["status"] == "unavailable"
                    and not item.get("covered_requirements")
                ],
            }
        )
    
    # Step 2: Process all tool calls together (both think_tool and ConductResearch)
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}
    plan = {item["id"]: item for item in state.get("research_plan", [])}

    plan_calls = [
        tool_call for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "CreateResearchPlan"
    ]
    for tool_call in plan_calls:
        if not state.get("research_plan"):
            plan = {
                dimension["dimension_id"]: {
                    "id": dimension["dimension_id"],
                    "subject": dimension["subject"],
                    "aspect": dimension["aspect"],
                    "deliverable": dimension["deliverable"],
                    "requirement": dimension["research_topic"],
                    "evidence_requirements": dimension["evidence_requirements"],
                    "queries": dimension.get("search_queries", []),
                    "status": "pending",
                    "source_ids": [],
                    "attempts": 0,
                    "covered_requirements": [],
                }
                for dimension in tool_call["args"]["dimensions"]
            }
            plan_message = "Research plan created."
        else:
            plan_message = "A research plan already exists."
        all_tool_messages.append(ToolMessage(
            content=(
                f"{plan_message} Delegate each pending dimension:\n"
                + "\n".join(
                    f"- {item['id']}: {item['requirement']} "
                    f"(requirements: {item['evidence_requirements']})"
                    for item in plan.values()
                )
            ),
            name="CreateResearchPlan",
            tool_call_id=tool_call["id"],
        ))
    
    # Handle think_tool calls (strategic reflection)
    think_tool_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "think_tool"
    ]
    
    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(ToolMessage(
            content=f"Reflection recorded: {reflection_content}",
            name="think_tool",
            tool_call_id=tool_call["id"]
        ))
    
    # Handle ConductResearch calls (research delegation)
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "ConductResearch"
    ]

    if not conduct_research_calls and not research_complete_tool_call:
        pending_item = next(
            (item for item in plan.values() if item["status"] == "pending"),
            None,
        )
        if pending_item:
            conduct_research_calls = [{
                "name": "ConductResearch",
                "id": f"automatic-{pending_item['id']}-{pending_item['attempts'] + 1}",
                "args": {
                    "dimension_id": pending_item["id"],
                    "subject": pending_item["subject"],
                    "aspect": pending_item["aspect"],
                    "deliverable": pending_item["deliverable"],
                    "research_topic": pending_item["requirement"],
                    "evidence_requirements": pending_item["evidence_requirements"],
                    "search_queries": pending_item["queries"],
                },
            }]

    if conduct_research_calls and not plan_calls:
        allowed_conduct_research_calls = []
        try:
            # Limit concurrent research units to prevent resource exhaustion
            allowed_conduct_research_calls = conduct_research_calls[:configurable.max_concurrent_research_units]
            overflow_conduct_research_calls = conduct_research_calls[configurable.max_concurrent_research_units:]
            for index, tool_call in enumerate(allowed_conduct_research_calls):
                args = tool_call["args"]
                dimension_id = args.get("dimension_id") or f"dimension-{len(plan) + index + 1:03}"
                args["dimension_id"] = dimension_id
                item = plan.setdefault(dimension_id, {
                    "id": dimension_id,
                    "subject": args.get("subject", args["research_topic"]),
                    "aspect": args.get("aspect", "requested evidence"),
                    "deliverable": args.get("deliverable", args["research_topic"]),
                    "requirement": args["research_topic"],
                    "evidence_requirements": args["evidence_requirements"],
                    "queries": args.get("search_queries", []),
                    "status": "pending",
                    "source_ids": [],
                    "attempts": 0,
                })
                item["attempts"] += 1
                item["status"] = "in_progress"
                logger.info(
                    "Researching dimension %s (attempt %d/%d)",
                    dimension_id, item["attempts"], MAX_DIMENSION_ATTEMPTS,
                )
            
            # Execute research tasks in parallel
            research_tasks = [
                researcher_subgraph.ainvoke({
                    "researcher_messages": [
                        HumanMessage(content=(
                            f"Research dimension {tool_call['args'].get('dimension_id', '')}:\n"
                            f"Subject: {tool_call['args'].get('subject', '')}\n"
                            f"Aspect: {tool_call['args'].get('aspect', '')}\n"
                            f"Deliverable: {tool_call['args'].get('deliverable', '')}\n"
                            f"{tool_call['args']['research_topic']}\n"
                            f"Evidence requirements: {tool_call['args']['evidence_requirements']}\n"
                            f"Targeted search queries: {tool_call['args'].get('search_queries', [])}"
                        ))
                    ],
                    "research_topic": tool_call["args"]["research_topic"],
                    "research_dimension_id": tool_call["args"].get("dimension_id", ""),
                    "evidence_requirements": tool_call["args"]["evidence_requirements"],
                    "run_id": state.get("run_id", ""),
                }, config) 
                for tool_call in allowed_conduct_research_calls
            ]
            
            tool_results = await asyncio.gather(*research_tasks, return_exceptions=True)

            # Create tool messages with research results
            for observation, tool_call in zip(tool_results, allowed_conduct_research_calls):
                if isinstance(observation, BaseException):
                    dimension_id = tool_call["args"].get("dimension_id", "")
                    error = (
                        f"Research dimension {dimension_id} failed: "
                        f"{type(observation).__name__}: {observation}"
                    )
                    logger.error(
                        error,
                        extra={"run_id": state.get("run_id", "unknown")},
                        exc_info=(
                            type(observation),
                            observation,
                            observation.__traceback__,
                        ),
                    )
                    plan[dimension_id]["status"] = (
                        "unavailable"
                        if plan[dimension_id]["attempts"] >= MAX_DIMENSION_ATTEMPTS
                        else "pending"
                    )
                    all_tool_messages.append(ToolMessage(
                        content=error,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    ))
                    update_payload.setdefault("research_errors", []).append(error)
                    continue
                dimension_id = observation.get(
                    "research_dimension_id",
                    tool_call["args"].get("dimension_id", ""),
                )
                raw_notes = "\n".join(observation.get("raw_notes", []))
                evidence_blocks = "\n".join(observation.get("evidence_blocks", []))
                coverage_notes = evidence_blocks or raw_notes
                if dimension_id in plan:
                    source_ids = sorted(
                        set(re.findall(r"\[?(S[0-9a-f]{6})\]?", raw_notes))
                    )
                    covered_requirements = sorted(_covered_requirement_numbers(
                        plan[dimension_id]["evidence_requirements"],
                        coverage_notes,
                        source_ids,
                    ))
                    plan[dimension_id]["covered_requirements"] = covered_requirements
                    plan[dimension_id]["status"] = (
                        "covered"
                        if set(covered_requirements)
                        >= set(range(1, len(plan[dimension_id]["evidence_requirements"]) + 1))
                        else "unavailable"
                        if plan[dimension_id]["attempts"] >= MAX_DIMENSION_ATTEMPTS
                        else "pending"
                    )
                    plan[dimension_id]["source_ids"] = source_ids
                    logger.info(
                        "Dimension %s: %s, requirements %d/%d, sources %d",
                        dimension_id,
                        plan[dimension_id]["status"],
                        len(covered_requirements),
                        len(plan[dimension_id]["evidence_requirements"]),
                        len(source_ids),
                    )
                    if not source_ids:
                        empty_error = (
                            f"Research dimension {dimension_id} returned no usable sources "
                            f"on attempt {plan[dimension_id]['attempts']}"
                        )
                        update_payload.setdefault("research_errors", []).append(empty_error)
                        all_tool_messages.append(ToolMessage(
                            content=empty_error,
                            name=tool_call["name"],
                            tool_call_id=tool_call["id"],
                        ))
                all_tool_messages.append(ToolMessage(
                    content=(
                        f"Research dimension {dimension_id} result:\n"
                        f"{observation.get('compressed_research', 'Error synthesizing research report: Maximum retries exceeded')}"
                    ),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                ))
            
            # Handle overflow research calls with error messages
            for overflow_call in overflow_conduct_research_calls:
                all_tool_messages.append(ToolMessage(
                    content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {configurable.max_concurrent_research_units} or fewer research units.",
                    name="ConductResearch",
                    tool_call_id=overflow_call["id"]
                ))
            
            # Aggregate raw notes from all research results
            raw_notes_concat = "\n".join([
                "\n".join(observation.get("raw_notes", [])) 
                for observation in tool_results
                if not isinstance(observation, BaseException)
            ])
            
            if raw_notes_concat:
                update_payload["raw_notes"] = [raw_notes_concat]
            evidence_blocks = [
                block
                for observation in tool_results
                if not isinstance(observation, BaseException)
                for block in observation.get("evidence_blocks", [])
            ]
            if evidence_blocks:
                update_payload["evidence_blocks"] = evidence_blocks
                
        except Exception as e:
            error = f"Research orchestration failed: {type(e).__name__}: {e}"
            logger.exception(error, extra={"run_id": state.get("run_id", "unknown")})
            for tool_call in allowed_conduct_research_calls:
                dimension_id = tool_call["args"].get("dimension_id", "")
                if dimension_id in plan:
                    plan[dimension_id]["status"] = (
                        "unavailable"
                        if plan[dimension_id]["attempts"] >= MAX_DIMENSION_ATTEMPTS
                        else "pending"
                    )
            all_tool_messages.append(ToolMessage(
                content=error,
                name="research_error",
                tool_call_id="research_error",
            ))
            update_payload["research_errors"] = [error]
    
    # Step 3: Return command with all tool results
    update_payload["supervisor_messages"] = all_tool_messages
    update_payload["research_plan"] = {
        "type": "override",
        "value": list(plan.values()),
    }
    return Command(
        goto="supervisor",
        update=update_payload
    ) 

# Supervisor Subgraph Construction
# Creates the supervisor workflow that manages research delegation and coordination
supervisor_builder = StateGraph(SupervisorState, context_schema=Configuration)

# Add supervisor nodes for research management
supervisor_builder.add_node("supervisor", supervisor)           # Main supervisor logic
supervisor_builder.add_node("supervisor_tools", supervisor_tools)  # Tool execution handler

# Define supervisor workflow edges
supervisor_builder.add_edge(START, "supervisor")  # Entry point to supervisor

# Compile supervisor subgraph for use in main workflow
supervisor_subgraph = supervisor_builder.compile()

async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    """Individual researcher that conducts focused research on specific topics.
    
    This researcher is given a specific research topic by the supervisor and uses
    available tools (search, think_tool, MCP tools) to gather comprehensive information.
    It can use think_tool for strategic planning between searches.
    
    Args:
        state: Current researcher state with messages and topic context
        config: Runtime configuration with model settings and tool availability
        
    Returns:
        Command to proceed to researcher_tools for tool execution
    """
    # Step 1: Load configuration and validate tool availability
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    
    # Get all available research tools (search, MCP, think_tool)
    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError(
            "No tools found to conduct research: Please configure either your "
            "search API or add MCP tools to your configuration."
        )
    
    # Step 2: Configure the researcher model with tools
    researcher_limit_param = "num_predict" if configurable.research_model.startswith("ollama:") else "max_tokens"
    research_model_config = {
        "model": configurable.research_model,
        researcher_limit_param: configurable.research_model_max_tokens,
        "tags": ["langsmith:nostream"]
    }
    
    # Prepare system prompt with MCP context if available
    researcher_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or "", 
        date=get_today_str()
    )
    
    # Configure model with tools, retry logic, and settings
    research_model = (
        configurable_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 3: Generate researcher response with system context
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    response = await research_model.ainvoke(messages)
    
    # Step 4: Update state and proceed to tool execution
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1
        }
    )

# Tool Execution Helper Function
async def execute_tool_safely(tool, args, config, run_id="unknown"):
    """Safely execute a tool with error handling."""
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        logger.exception(
            "Tool %s failed",
            getattr(tool, "name", "unknown"),
            extra={"run_id": run_id},
        )
        return f"Error executing tool: {str(e)}"


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    """Execute tools called by the researcher, including search tools and strategic thinking.
    
    This function handles various types of researcher tool calls:
    1. think_tool - Strategic reflection that continues the research conversation
    2. Search tools (searxng_search, web_search) - Information gathering
    3. MCP tools - External tool integrations
    4. ResearchComplete - Signals completion of individual research task
    
    Args:
        state: Current researcher state with messages and iteration count
        config: Runtime configuration with research limits and tool settings
        
    Returns:
        Command to either continue research loop or proceed to compression
    """
    # Step 1: Extract current state and check early exit conditions
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]
    
    # Early exit if no tool calls were made
    if not most_recent_message.tool_calls:
        return Command(goto="compress_research")
    
    # Step 2: Handle other tool calls (search, MCP tools, etc.)
    tools = await get_all_tools(config)
    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool 
        for tool in tools
    }
    
    # Execute all tool calls in parallel
    tool_calls = most_recent_message.tool_calls
    tool_execution_tasks = [
        execute_tool_safely(
            tools_by_name[tool_call["name"]],
            tool_call["args"],
            config,
            state.get("run_id", "unknown"),
        )
        for tool_call in tool_calls
    ]
    observations = await asyncio.gather(*tool_execution_tasks)
    
    # Create tool messages from execution results
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) 
        for observation, tool_call in zip(observations, tool_calls)
    ]
    
    # Step 3: Check late exit conditions (after processing tools)
    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    if exceeded_iterations or research_complete_called:
        # End research and proceed to compression
        return Command(
            goto="compress_research",
            update={"researcher_messages": tool_outputs}
        )
    
    # Continue research loop with tool results
    return Command(
        goto="researcher",
        update={"researcher_messages": tool_outputs}
    )

async def compress_research(state: ResearcherState, config: RunnableConfig):
    """Compress and synthesize research findings into a concise, structured summary.
    
    This function takes all the research findings, tool outputs, and AI messages from
    a researcher's work and distills them into a clean, comprehensive summary while
    preserving all important information and findings.
    
    Args:
        state: Current researcher state with accumulated research messages
        config: Runtime configuration with compression model settings
        
    Returns:
        Dictionary containing compressed research summary and raw notes
    """
    logger.info("Compressing findings for dimension %s", state.get("research_dimension_id", "unknown"))
    # Step 1: Configure the compression model
    configurable = Configuration.from_runnable_config(config)
    compression_limit_param = "num_predict" if configurable.compression_model.startswith("ollama:") else "max_tokens"
    synthesizer_model = configurable_model.with_config({
        "model": configurable.compression_model,
        compression_limit_param: configurable.compression_model_max_tokens,
        "tags": ["langsmith:nostream"]
    })
    
    # Step 2: Prepare messages for compression
    researcher_messages = state.get("researcher_messages", [])
    
    # Add instruction to switch from research mode to compression mode
    requirements = state.get("evidence_requirements", [])
    evidence_checklist = "\n".join(
        f"{index}. {requirement}"
        for index, requirement in enumerate(requirements, start=1)
    )
    researcher_messages.append(HumanMessage(content=(
        f"{compress_research_simple_human_message}\n\n"
        "Return one block for every requirement using exactly this format:\n"
        "Requirement 1: <requirement>\n"
        "Evidence: <source-backed evidence or unavailable>\n"
        "Sources: [Sxxxxxx]\n\n"
        f"Evidence requirements:\n{evidence_checklist}"
    )))
    
    # Step 3: Attempt compression with retry logic for token limit issues
    synthesis_attempts = 0
    max_attempts = 3
    
    while synthesis_attempts < max_attempts:
        try:
            # Create system prompt focused on compression task
            compression_prompt = compress_research_system_prompt.format(date=get_today_str())
            messages = [SystemMessage(content=compression_prompt)] + researcher_messages
            
            # Execute compression
            response = await synthesizer_model.ainvoke(messages)
            
            # Extract raw notes from all tool and AI messages
            raw_notes_content = "\n".join([
                str(message.content) 
                for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
            ] + [str(response.content)])
            
            # Return successful compression result
            return {
                "compressed_research": str(response.content),
                "raw_notes": [raw_notes_content],
                "evidence_blocks": [str(response.content)],
                "research_dimension_id": state.get("research_dimension_id", ""),
            }
            
        except Exception as e:
            synthesis_attempts += 1
            logger.warning(
                "Research compression attempt %d failed",
                synthesis_attempts,
                extra={"run_id": state.get("run_id", "unknown")},
                exc_info=True,
            )
            
            # Handle token limit exceeded by removing older messages
            if is_token_limit_exceeded(e, configurable.research_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                continue
            
            # For other errors, continue retrying
            continue
    
    # Step 4: Return error result if all attempts failed
    raw_notes_content = "\n".join([
        str(message.content) 
        for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
    ])
    
    return {
        "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
        "raw_notes": [raw_notes_content],
        "evidence_blocks": [],
        "research_dimension_id": state.get("research_dimension_id", ""),
    }

# Researcher Subgraph Construction
# Creates individual researcher workflow for conducting focused research on specific topics
researcher_builder = StateGraph(
    ResearcherState, 
    output_schema=ResearcherOutputState,
    context_schema=Configuration,
)

# Add researcher nodes for research execution and compression
researcher_builder.add_node("researcher", researcher)                 # Main researcher logic
researcher_builder.add_node("researcher_tools", researcher_tools)     # Tool execution handler
researcher_builder.add_node("compress_research", compress_research)   # Research compression

# Define researcher workflow edges
researcher_builder.add_edge(START, "researcher")           # Entry point to researcher
researcher_builder.add_edge("compress_research", END)      # Exit point after compression

# Compile researcher subgraph for parallel execution by supervisor
researcher_subgraph = researcher_builder.compile()

async def final_report_generation(state: AgentState, config: RunnableConfig):
    """Generate the final comprehensive research report with retry logic for token limits.
    
    This function takes all collected research findings and synthesizes them into a 
    well-structured, comprehensive final report using the configured report generation model.
    
    Args:
        state: Agent state containing research findings and context
        config: Runtime configuration with model settings and API keys
        
    Returns:
        Dictionary containing the final report and cleared state
    """
    # Step 1: Extract research findings and prepare state cleanup
    notes = state.get("notes", [])
    raw_notes = state.get("raw_notes", [])
    research_plan = state.get("research_plan", [])
    pending = _pending_dimensions(research_plan)
    if pending:
        message = (
            "Research incomplete: pending dimensions were not resolved: "
            + ", ".join(item["id"] for item in pending)
        )
        logger.warning("%s", message)
        return {
            "final_report": message,
            "messages": [AIMessage(content=message)],
            "report_dimension_ids": [],
            "report_evidence_markers": [],
        }
    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = _findings_for_report(
        notes,
        state.get("evidence_blocks", []),
        raw_notes,
    )
    logger.info(
        "Writing report: %d dimensions covered, %d unavailable",
        sum(item.get("status") == "covered" for item in research_plan),
        sum(item.get("status") == "unavailable" for item in research_plan),
    )
    
    # Step 2: Configure the final report generation model
    configurable = Configuration.from_runnable_config(config)
    writer_limit_param = "num_predict" if configurable.final_report_model.startswith("ollama:") else "max_tokens"
    writer_model_config = {
        "model": configurable.final_report_model,
        writer_limit_param: configurable.final_report_model_max_tokens,
        "tags": ["langsmith:nostream"]
    }
    
    # Step 3: Attempt report generation with token limit retry logic
    max_retries = 3
    current_retry = 0
    findings_token_limit = None
    
    while current_retry <= max_retries:
        try:
            # Create comprehensive prompt with all research context
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                research_plan="\n".join(
                    f"- {item['id']}: {item['requirement']} "
                    f"(subject: {item['subject']}; aspect: {item['aspect']}; "
                    f"evidence requirements: {item['evidence_requirements']}; "
                    f"status: {item['status']}, sources: {', '.join(item['source_ids']) or 'none'})"
                    for item in research_plan
                ),
                dimension_markers="\n".join(
                    f"<!-- dimension: {item['id']} -->"
                    for item in research_plan
                ),
                coverage_markers="\n".join(
                    f"<!-- evidence: {item['id']}:{number} -->"
                    for item in research_plan
                    for number in item["covered_requirements"]
                ),
                date=get_today_str()
            )
            
            # Generate the final report
            final_report = await configurable_model.with_config(writer_model_config).ainvoke([
                HumanMessage(content=final_report_prompt)
            ])
            
            report_text = str(final_report.content)
            missing_requirements = _missing_report_requirements(report_text, research_plan)
            missing_dimensions = _missing_report_dimensions(report_text, research_plan)
            if missing_requirements or missing_dimensions:
                logger.info(
                    "Repairing report: %d missing dimensions, %d missing evidence markers",
                    len(missing_dimensions), len(missing_requirements),
                )
                missing_plan = "\n".join(
                    f"- {item['id']}: {item['requirement']} "
                    f"(subject: {item['subject']}; aspect: {item['aspect']}; "
                    f"evidence requirements: {item['evidence_requirements']}; "
                    f"sources: {', '.join(item['source_ids']) or 'none'})"
                    for item in research_plan
                    if item["id"] in missing_dimensions
                    or any(
                        marker.startswith(f"{item['id']}:")
                        for marker in missing_requirements
                    )
                )
                repair_prompt = (
                    "Return only new sections for the missing dimensions below. Do not "
                    "rewrite or summarize the existing report. Use the supplied findings "
                    "and source IDs; never invent citations. Include every requested "
                    "dimension and evidence marker.\n\nMissing plan:\n"
                    f"{missing_plan}\n\nMissing dimensions: "
                    f"{', '.join(missing_dimensions) or 'none'}; missing evidence markers: "
                    f"{', '.join(missing_requirements) or 'none'}\n\n"
                    f"Findings:\n{findings}\n\nExisting report:\n{report_text}"
                )
                repaired_report = await configurable_model.with_config(writer_model_config).ainvoke([
                    HumanMessage(content=repair_prompt)
                ])
                report_text = _merge_report_repair(
                    report_text,
                    str(repaired_report.content),
                    research_plan,
                )
                repaired_missing_requirements = _missing_report_requirements(
                    report_text,
                    research_plan,
                )
                repaired_missing_dimensions = _missing_report_dimensions(
                    report_text,
                    research_plan,
                )
                if repaired_missing_requirements or repaired_missing_dimensions:
                    error = (
                        "Report repair failed coverage validation: "
                        f"missing dimensions={repaired_missing_dimensions or 'none'}, "
                        f"missing evidence={repaired_missing_requirements or 'none'}"
                    )
                    logger.error(error, extra={"run_id": state.get("run_id", "unknown")})
                    return {
                        "final_report": f"Error generating final report: {error}",
                        "messages": [AIMessage(content=error)],
                    }
            report_dimension_ids = sorted(set(re.findall(
                r"<!-- dimension: ([^ ]+) -->",
                report_text,
            )))
            report_evidence_markers = sorted(set(re.findall(
                r"<!-- evidence: ([^ ]+:\d+) -->",
                report_text,
            )))
            report_text = re.sub(
                r"<!-- (?:evidence: [^>]+|dimension: [^>]+) -->\s*",
                "",
                report_text,
            )
            plan_errors = _research_plan_errors(research_plan)
            if plan_errors:
                error = "Research plan validation failed: " + "; ".join(plan_errors)
                logger.error(error, extra={"run_id": state.get("run_id", "unknown")})
                return {
                    "final_report": f"Error generating final report: {error}",
                    "messages": [AIMessage(content=error)],
                    **cleared_state
                }
            final_report.content = finalize_citations(report_text, raw_notes)
            if not _has_source_backed_citations(final_report.content, raw_notes):
                error = (
                    "Final report omitted citations despite retrieved source evidence. "
                    "The report was rejected to avoid returning unsupported claims."
                )
                logger.error(error, extra={"run_id": state.get("run_id", "unknown")})
                return {
                    "final_report": f"Error generating final report: {error}",
                    "messages": [AIMessage(content=error)],
                    **cleared_state
                }
            logger.info("Final report generated")
            return {
                "final_report": final_report.content, 
                "messages": [final_report],
                "report_dimension_ids": report_dimension_ids,
                "report_evidence_markers": report_evidence_markers,
                **cleared_state
            }
            
        except Exception as e:
            # Handle token limit exceeded errors with progressive truncation
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1
                
                if current_retry == 1:
                    # First retry: determine initial truncation limit
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}",
                            "messages": [AIMessage(content="Report generation failed due to token limits")],
                            **cleared_state
                        }
                    # Use 4x token limit as character approximation for truncation
                    findings_token_limit = model_token_limit * 4
                else:
                    # Subsequent retries: reduce by 10% each time
                    findings_token_limit = int(findings_token_limit * 0.9)
                
                # Truncate findings and retry
                findings = findings[:findings_token_limit]
                continue
            else:
                # Non-token-limit error: return error immediately
                logger.exception(
                    "Final report generation failed",
                    extra={"run_id": state.get("run_id", "unknown")},
                )
                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [AIMessage(content="Report generation failed due to an error")],
                    **cleared_state
                }
    
    # Step 4: Return failure result if all retries exhausted
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [AIMessage(content="Report generation failed after maximum retries")],
        **cleared_state
    }

# Main Deep Researcher Graph Construction
# Creates the complete deep research workflow from user input to final report
deep_researcher_builder = StateGraph(
    AgentState, 
    input_schema=AgentInputState,
    context_schema=Configuration,
)

# Add main workflow nodes for the complete research process
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)           # User clarification phase
deep_researcher_builder.add_node("write_research_brief", write_research_brief)     # Research planning phase
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)       # Research execution phase
deep_researcher_builder.add_node("final_report_generation", final_report_generation)  # Report generation phase

# Define main workflow edges for sequential execution
deep_researcher_builder.add_edge(START, "clarify_with_user")                       # Entry point
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation") # Research to report
deep_researcher_builder.add_edge("final_report_generation", END)                   # Final exit point

# Compile the complete deep researcher workflow
deep_researcher = deep_researcher_builder.compile()