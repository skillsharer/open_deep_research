"""Graph state definitions and data structures for the Deep Research agent."""

import operator
from typing import Annotated, Literal, Optional

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import TypedDict


###################
# Structured Outputs
###################
class ConductResearch(BaseModel):
    """Call this tool to conduct research on a specific topic."""
    dimension_id: str = Field(
        description="Stable ID of the research-plan dimension this task covers, such as dimension-001.",
    )
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )
    evidence_requirements: list[str] = Field(
        description="Atomic facts this task must establish, one requirement per item.",
    )
    search_queries: list[str] = Field(
        default_factory=list,
        description="Targeted search queries for this dimension.",
    )

class ResearchPlanDimension(BaseModel):
    """One dimension in the complete research plan."""
    dimension_id: str
    subject: str = Field(description="The entity, object, period, factor, or facet being researched.")
    aspect: str = Field(description="The requested attribute or question facet for this subject.")
    deliverable: str = Field(description="The specific answer unit this research must provide.")
    research_topic: str
    evidence_requirements: list[str]
    search_queries: list[str] = Field(default_factory=list)

    @field_validator("dimension_id", "subject", "aspect", "deliverable", "research_topic")
    @classmethod
    def require_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("research plan fields must not be empty")
        return value

    @field_validator("evidence_requirements", "search_queries")
    @classmethod
    def require_unique_items(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values if value.strip()]
        if not cleaned or len(cleaned) != len(set(cleaned)):
            raise ValueError("research plan lists must contain unique non-empty items")
        return cleaned

class CreateResearchPlan(BaseModel):
    """Create the complete research plan before delegating research."""
    dimensions: list[ResearchPlanDimension]

    @model_validator(mode="after")
    def require_unique_dimensions(self):
        ids = [dimension.dimension_id for dimension in self.dimensions]
        subjects = [
            (dimension.subject.casefold(), dimension.aspect.casefold())
            for dimension in self.dimensions
        ]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("research plan dimension IDs must be unique")
        if len(subjects) != len(set(subjects)):
            raise ValueError("research plan dimensions must be unique by subject and aspect")
        if any(not dimension.search_queries for dimension in self.dimensions):
            raise ValueError("each research plan dimension must have a search query")
        return self

class ResearchComplete(BaseModel):
    """Call this tool to indicate that the research is complete."""

class Summary(BaseModel):
    """Research summary with key findings."""
    
    summary: str
    key_excerpts: str

class ClarifyWithUser(BaseModel):
    """Model for user clarification requests."""
    
    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )

class ResearchQuestion(BaseModel):
    """Research question and brief for guiding research."""
    
    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )


###################
# State Definitions
###################

class ResearchPlanItem(TypedDict):
    id: str
    subject: str
    aspect: str
    deliverable: str
    requirement: str
    evidence_requirements: list[str]
    queries: list[str]
    status: Literal["pending", "in_progress", "covered", "unavailable"]
    source_ids: list[str]
    attempts: int
    covered_requirements: list[int]

def override_reducer(current_value, new_value):
    """Reducer function that allows overriding values in state."""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)
    
class AgentInputState(MessagesState):
    """Initial messages plus an optional correlation identifier."""
    run_id: str

class AgentState(MessagesState):
    """Main agent state containing messages and research data."""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]
    run_id: str
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    research_errors: Annotated[list[str], override_reducer] = []
    evidence_blocks: Annotated[list[str], override_reducer] = []
    research_plan: Annotated[list[ResearchPlanItem], override_reducer] = []
    report_dimension_ids: list[str] = []
    report_evidence_markers: list[str] = []
    final_report: str

class SupervisorState(TypedDict):
    """State for the supervisor that manages research tasks."""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    run_id: str
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: int = 0
    raw_notes: Annotated[list[str], override_reducer] = []
    research_errors: Annotated[list[str], override_reducer] = []
    evidence_blocks: Annotated[list[str], override_reducer] = []
    research_plan: Annotated[list[ResearchPlanItem], override_reducer] = []

class ResearcherState(TypedDict):
    """State for individual researchers conducting research."""
    
    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    run_id: str
    tool_call_iterations: int = 0
    research_topic: str
    research_dimension_id: str
    evidence_requirements: list[str]
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherOutputState(BaseModel):
    """Output state from individual researchers."""
    
    compressed_research: str
    research_dimension_id: str
    raw_notes: Annotated[list[str], override_reducer] = []
    evidence_blocks: list[str] = []