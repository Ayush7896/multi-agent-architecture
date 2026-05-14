import operator
import warnings
from typing import Annotated, TypedDict, List, Literal
from pydantic import BaseModel, ConfigDict

# WHY add chat_id to State?
# State flows through every node. Adding chat_id here means
# any agent can call load_chat_history(state["chat_id"]) in Phase 2.
# FastAPI sets it once at the start; agents read it, never modify it.

class PlannerState(TypedDict):
    user_input: str
    chat_id: str     # state used for persistence
    current_thought: str
    plan: Annotated[list[str], operator.add]
    next_agent: str
    research_findings: str    # New: researcher writes here, planner reads
    final_answer: str


# ── WHY model_config with arbitrary_types_allowed? ───────────────────────────
# LangGraph checkpointing serializes ALL values that pass through the graph,
# including the intermediate structured output objects (OrchestratorOutput,
# PlannerOutput) before they are unpacked into the state dict.
#
# Pydantic v2 generates a UserWarning:
#   "Expected `none` but got `OrchestratorOutput`"
# because LangGraph's serializer schema expects None but receives the model instance.
#
# model_config = ConfigDict(arbitrary_types_allowed=True) tells Pydantic:
#   "Don't raise/warn if an unexpected type passes through serialization."
# This is the official Pydantic v2 fix for LangGraph interop — purely cosmetic,
# no functional change. The values are still correctly extracted into state.
# ─────────────────────────────────────────────────────────────────────────────

class OrchestratorOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    thought:    str
    next_agent: Literal["researcher", "planner", "end"]
    plan:       List[str]


class PlannerOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    final_answer: str


# Suppress the residual Pydantic serializer warning that can still surface
# from LangGraph's internal checkpoint machinery even after the above fix.
# This is a known LangGraph + Pydantic v2 interop issue — tracked upstream.
warnings.filterwarnings(
    "ignore",
    message=".*Pydantic serializer warnings.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*Expected `none` but got.*",
    category=UserWarning,
)