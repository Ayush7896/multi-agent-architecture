import json
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, SystemMessage
from agents.schemas import PlannerState, OrchestratorOutput
from agents.model import model
from langchain_core.prompts import ChatPromptTemplate
from agents.planner import planner_agent

def orchestrator_agent(state: PlannerState):
    orchestrator_prompt = ChatPromptTemplate.from_messages(
      [
         ("system", """ You are an advanced orchestrator.
         """),
         ("human","{user_input}")
      ]
    )
    structured_model = model.with_structured_output(OrchestratorOutput)
    messages = orchestrator_prompt.format_messages(
      user_input = state["user_input"]
    )
    response = structured_model.invoke(messages)

    return {
        "current_thought": response.thought,
        "plan": response.plan,
        "next_agent": response.next_agent
    }


graph = StateGraph(PlannerState)
graph.add_node("orchestrator",orchestrator_agent)
graph.add_node("planner",planner_agent)

def route(state:PlannerState):
    return state["next_agent"]

graph.add_edge(START, "orchestrator")
graph.add_conditional_edges(
    "orchestrator",
    route,
    {
        "planner":"planner",
        "end": END
    }
)
graph.add_edge("planner",END)


workflow = graph.compile()

