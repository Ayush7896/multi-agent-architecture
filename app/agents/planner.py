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
    # adding retries logic
    for attempt in range(3):
        try:
            response = structured_model.invoke(messages)
            if not response.final_answer or len(response.final_answer.strip()) == 0:
                raise ValueError("Empty answer")
            return {
                "final_answer": response.final_answer
                }
        except Exception as e:
            print(f"Retry {attempt + 1} failed:",e)

    
    return {
        "final_answer": "Failed after retries"
    }