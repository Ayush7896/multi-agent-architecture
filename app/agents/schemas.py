import operator
from typing import Annotated, TypedDict,List, Literal
from pydantic import BaseModel

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
    final_answer: str

class OrchestratorOutput(BaseModel):
    thought: str
    next_agent: Literal["planner","end"]
    plan: List[str]

class PlannerOutput(BaseModel):
    final_answer: str