# """
# researcher.py

# WHAT: Fetches real-time data from the internet using Tavily search.
# WHY:  Planner was generating hallucinated hotel names, prices, opening hours.
#       Researcher grounds the answer in real current facts.
# HOW:  create_react_agent = LangGraph's built-in ReAct loop.
#       ReAct = Reason, Act, Observe, Reason again.
#       The LLM reasons about what to search, calls Tavily, reads results,
#       decides if it needs more searches, then writes a findings summary.

# WHAT IS create_react_agent?
#   It is a pre-built LangGraph agent that handles the tool-calling loop
#   for you. Without it you would write:
#     while True:
#         response = llm.invoke(messages)
#         if response has tool_call: run tool, append result, continue
#         else: break
#   create_react_agent does exactly that loop — you just give it tools.
# """

import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from agents.model import model
from agents.schemas import PlannerState
from db.database import load_chat_history

load_dotenv()

# ── The Tavily tool ──────────────────────────────────────────────────────────
# max_results=5 → top 5 web results per search query
# The LLM can call this tool MULTIPLE TIMES per researcher_agent invocation
# e.g. once for "best hotels Paris 2025", once for "Paris attractions itinerary"

tavily_tool = TavilySearchResults(max_results = 5)

def researcher_agent(state: PlannerState) -> dict:
    """
    Receives State from orchestrator.
    Reads state["plan"] — the list of things to research.
    Calls create_react_agent which runs the ReAct loop.
    Writes state["research_findings"] — real data for the planner.
    Also sets state["next_agent"] = "planner" to continue the graph.

    WHY does researcher set next_agent?
      After researcher runs, the graph needs to know where to go next.
      We always go to planner after research — so researcher sets it.
      This means route() in orchestrator.py also handles "planner"
      as a value that can come from orchestrator OR researcher.
    """
    # Load conversation context — researcher needs to know what's
    # already been planned in past turns (Phase 2 Memory)

    history = load_chat_history(state["chat_id"], limit = 6)

    # Build the research task from orchestrator's plan
    plan_text = "\n".join(state["plan"])

    # System Prompt - tells the LLM what is job is
    system = SystemMessage(
        content = """  
                You are a research agent. Your job is to search the internet
                for accurate, current information to help plan travel itineraries.

                For each research task:
                1. Search for specific, actionable data (prices, opening hours, ratings)
                2. Search multiple queries if needed for complete coverage
                3. Write a structured findings summary with source-backed facts

                Be specific. Include actual prices, actual names, actual hours.
                Do not make up any information — only use what you find in searches.
                  """
    )
    # Build the message that kicks off the ReACT loop
    research_request = HumanMessage(
        content = f""" 
                Research the following topics thoroughly:
                {plan_text}
                User content (conversation_history):
                {chr(10).join([f"{m['role']}: {m['content']}" for m in history])}
                Search for current, accurate information and compile your findings
                """
    )
    # ── create_react_agent — the heart of tool calling ───────────────────────
    # model  = your LLM (OpenAI/Anthropic/etc)
    # tools  = list of tools the LLM can call — just Tavily here,
    #          but you can add more: weather API, flight search, maps, etc.
    # state_modifier = prepends the system message to every LLM call in the loop
    #
    # The loop that runs internally:
    #   1. LLM receives [system, research_request]
    #   2. LLM responds: "I'll search for X" + tool_call(query="X")
    #   3. Tavily executes, returns 5 results
    #   4. Results appended as ToolMessage
    #   5. LLM decides: search again or write final answer
    #   6. Repeats until LLM writes a plain text response (no tool call)
    
    react_agent = create_react_agent(
        model = model,
        tools = [tavily_tool],
        state_modifier = system
    )
    # Invoke the ReAct agent — this runs the FULL loop until LLM stops calling tools
    result = react_agent.invoke({
        "messages": [research_request]
    })
    # The last message is always the LLM's final text response
    # (no tool call = it's done researching)
    research_findings = result["messages"][-1].content

    print(f"[Researcher] Found {len(result['messages'])} messages in ReAct loop")
    print(f"[Researcher] Findings preview: {research_findings[:200]}...")


    return {
        "research_findings": research_findings,
        "next_agent":        "planner"   # always go to planner after research
    }


