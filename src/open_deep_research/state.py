"""Graph state definitions and data structures for the Deep Research agent."""

import operator
from typing import Annotated, Literal, Optional

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
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

class CreateResearchPlan(BaseModel):
    """Create the complete research plan before delegating research."""
    dimensions: list[ResearchPlanDimension]

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
    status: Literal["pending", "covered", "unavailable"]
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
    """InputState is only 'messages'."""

class AgentState(MessagesState):
    """Main agent state containing messages and research data."""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    research_plan: Annotated[list[ResearchPlanItem], override_reducer] = []
    final_report: str

class SupervisorState(TypedDict):
    """State for the supervisor that manages research tasks."""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: int = 0
    raw_notes: Annotated[list[str], override_reducer] = []
    research_plan: Annotated[list[ResearchPlanItem], override_reducer] = []

class ResearcherState(TypedDict):
    """State for individual researchers conducting research."""
    
    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
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