import os
from langgraph.graph import StateGraph, START, END
from db.database import load_chat_history
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from agents.schemas import PlannerState, OrchestratorOutput
from agents.model import model
from langchain_core.prompts import ChatPromptTemplate
from agents.planner import planner_agent
from dotenv import load_dotenv
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

load_dotenv()

# Phase 2 (Memory) + Phase 3 (Checkpointer)
# Phase 2 changes:
#   - Import load_chat_history
#   - Build messages list manually with history injection
#   - Shorter history (limit=6) — orchestrator needs less context than planner

# Phase 3 changes:
#   - Import AsyncPostgresSaver, AsyncConnectionPool
#   - Build DB_URI for psycopg3 (different from psycopg2 used in database.py)
#   - Add create_workflow() async function — compiles graph WITH checkpointer
#   - Remove module-level workflow = graph.compile() — main.py owns compilation now

# WHY two DB drivers?
#   database.py  → psycopg2 (sync)  → YOUR chats/messages tables
#   orchestrator → psycopg3 (async) → LangGraph's checkpoint tables
#   They connect to the SAME PostgreSQL database but use different drivers.
#   psycopg2 is sync and simple — fine for your custom tables.
#   AsyncPostgresSaver requires psycopg3 (the `psycopg` package) — LangGraph's requirement.
#   They coexist without conflict.


# ── Phase 3: DB_URI for psycopg3 ────────────────────────────────────────────
# psycopg3 uses a connection string, not a dict like psycopg2.
# Format: postgresql://user:password@host:port/dbname
# This connects to the SAME database as database.py — different driver, same DB.
DB_URI = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)


# ── orchestrator_agent — Phase 2 Memory added ───────────────────────────────
def orchestrator_agent(state: PlannerState):
    """
    Orchestrator now knows about conversation history.

    WHY inject history into orchestrator too?
      If a user says "add Rome to that trip", the orchestrator needs to know
      "that trip" means the Paris trip from Turn 1.
      Without history, it would try to plan Rome as a standalone request.
      With history (limit=6, ~3 turns), it understands the session context.

    WHY fewer messages (limit=6) than planner (limit=10)?
      Orchestrator's job is routing and planning — it doesn't need
      the full detailed itinerary text. Just enough to understand the session.
      Planner creates the detailed answer so it needs more context.
    """
    history = load_chat_history(state["chat_id"], limit=6)    # Phase 2

    messages = [
        SystemMessage(content="""
            You are an advanced orchestrator.
            You analyze user requests and decide which specialized agent handles them.
            You are aware of the conversation history provided.
            Use it to understand follow-up requests and running context.
        """)
    ]

    # Inject history — same pattern as planner
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    # Current user input — what the user just sent
    messages.append(HumanMessage(content=state["user_input"]))
   

    structured_model = model.with_structured_output(OrchestratorOutput)
 
    # adding retries logic
    for attempt in range(3):
        try:
            response = structured_model.invoke(messages)
            if response.next_agent not in ["planner","end"]:
                raise ValueError("Invalid agent")
            return {
            "current_thought": response.thought,
            "plan": response.plan,
            "next_agent": response.next_agent
            }
        except Exception as e:
            print(f"Retry {attempt + 1} failed:",e)

    return {
        "next_agent": "end",
        "final_answer": "Failed after retries"
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

# ── Phase 3: Async workflow factory ─────────────────────────────────────────

async def create_workflow():
    """
    Compiles the graph WITH a PostgreSQL checkpointer.

    WHY async?
      AsyncPostgresSaver uses psycopg3's async connection pool.
      Async means FastAPI can handle other requests while waiting
      for the DB — no thread blocking.

    WHY AsyncConnectionPool and not a single connection?
      A single connection handles one query at a time.
      FastAPI serves multiple requests simultaneously.
      Pool size 10 = up to 10 concurrent graph executions.
      Without a pool, requests queue behind each other.

    WHY autocommit=True?
      LangGraph's checkpointer manages its own transactions.
      If you let psycopg3 auto-wrap in a transaction, it conflicts
      with LangGraph's own commit/rollback calls.

    WHY prepare_threshold=0?
      Prepared statements cache query plans on a specific connection.
      With a connection pool, the same query might run on different
      connections — caching breaks. Setting 0 disables it.

    WHY checkpointer.setup()?
      Creates LangGraph's three tables IF they don't exist:
        checkpoints       — one row per node execution
        checkpoint_writes — intermediate write buffer
        checkpoint_blobs  — binary State snapshots
      IF NOT EXISTS = safe to call every startup, exactly like your create_tables().
      You NEVER write SQL against these tables yourself.   
    """
    pool = AsyncConnectionPool(
        conninfo=DB_URI,
        max_size = 10,
        kwargs = {"autocommit": True, "prepare_threshold": 0}
    )

    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()  # langgraph creates its own tables here

    workflow = graph.compile(checkpointer=checkpointer)
    print("[LangGraph] Checkpointer ready — LangGraph tables created/verified")
    return workflow


