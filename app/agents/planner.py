from langchain_core.prompts import ChatPromptTemplate
from agents.model import model
from agents.schemas import PlannerState, PlannerOutput
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from db.database import load_chat_history

# Memory added
# What changes from Phase 1:
#   - Import load_chat_history and message types
#   - Build messages list manually instead of using ChatPromptTemplate
#   - Inject history BETWEEN system message and current input
#   - Structured model invoked on the full messages list

# WHY manual messages list instead of ChatPromptTemplate?
#   ChatPromptTemplate is for fixed, templated messages.
#   When you dynamically inject a variable number of history messages,
#   a plain Python list is cleaner and more readable.
#   You can still use ChatPromptTemplate for the system message.
   
def planner_agent(state: PlannerState):
    # ── Phase 2: Load conversation history from YOUR PostgreSQL ─────────
    # This is Persistence → Memory: reading from DB, injecting into prompt
    # limit=10 means last 10 messages (5 turns of user+ai)
    # WHY 10 and not all? Token limits. GPT-4o = 128k tokens.
    # 1 message ≈ 200-500 tokens. 10 messages ≈ safe budget.
    # In production: use tiktoken to count exact tokens, not message count.

    history = load_chat_history(state["chat_id"], limit = 10)
    # ── Build messages list — THIS is the Memory injection ──────────────
    messages = [
        SystemMessage(
            content = """  
                    You are a travel planner agent.
                    You create detailed, day-by-day travel itineraries.
                    You are aware of the full conversation history provided.
                    If the user references a previous trip, build upon it.
                    """
        )
    ]
    # Inject history — past turns become part of the LLM's context
    # WHY this order? Chronological oldest→newest so LLM reads like a chat.
    # load_chat_history returns oldest→newest (ORDER BY timestamp ASC)
    for msg in history:
        if msg["role"] == 'user':
            messages.append(HumanMessage(content = msg["content"]))
        else:
            messages.append(AIMessage(content = msg["content"]))

    # Current request — the orchestrator's plan, not raw user_input
    # The orchestrator already distilled user_input into structured plan steps
    messages.append(
    HumanMessage(content=f"""
        Plan to execute:
        {chr(10).join(state["plan"])}

        Research findings (real current data from internet):
        {state.get("research_findings", "No research data available.")}

        Create a detailed, accurate itinerary using the research above.
        Reference specific names, prices, and details from the research.
        """)
    )

    structured_model = model.with_structured_output(PlannerOutput)

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