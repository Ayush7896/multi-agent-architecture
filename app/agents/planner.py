from langchain_core.prompts import ChatPromptTemplate
from agents.model import model
from agents.schemas import PlannerState, PlannerOutput
import json

def planner_agent(state: PlannerState):

    planner_prompt = ChatPromptTemplate(
        [
            ("system", """
            You are an planner agent and you plan the itenearies
             """),
            ("human", "{plan_input}")
        ]
    )
    
    structured_model = model.with_structured_output(PlannerOutput)

    messages = planner_prompt.format_messages(
        plan_input = "\n".join(state["plan"])
    )
    response = structured_model.invoke(messages)
    return {
        "final_answer": response.final_answer
    }