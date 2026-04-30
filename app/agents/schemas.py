import operator
from typing import Annotated, TypedDict,List, Literal
from pydantic import BaseModel


class PlannerState(TypedDict):
    user_input: str
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