# Phase 1
# Persistence lives HERE — in the FastAPI layer, outside LangGraph.
# LangGraph knows nothing about your chats/messages tables.

# Request lifecycle:
#   1. Receive request
#   2. Create or reuse chat session
#   3. Save user message  ← Persistence
#   4. Run workflow       ← LangGraph (State lives here)
#   5. Save AI response   ← Persistence
#   6. Return response

# """ 
# Phase 3 (Async + Checkpointer)

# What changes from Phase 1:
#   - Import create_workflow instead of workflow directly
#   - Global `workflow` variable set in lifespan (async init required)
#   - Endpoint becomes async (was already async, but now ainvoke)
#   - workflow.ainvoke() instead of workflow.invoke()
#   - config dict with thread_id passed to ainvoke

# WHY global workflow variable?
#   create_workflow() is async — can't run at import time.
#   It must run inside an async context (lifespan).
#   global workflow = set once at startup, used on every request.

# WHY thread_id = chat_id?
#   thread_id is LangGraph's key for grouping checkpoints.
#   All checkpoints from the same thread_id belong to one session.
#   Using chat_id as thread_id ties LangGraph's session to YOUR session.
#   Same chat = same thread = LangGraph can resume/replay that conversation's
#   State from any checkpoint. One ID governs both layers.
# """



from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from db.database import create_tables, create_chat,save_message, load_chat_history
from contextlib import asynccontextmanager
from agents.orchestrator import create_workflow



# ─────────────────────────────────────────────
# LIFESPAN
# Replaces @app.on_event("startup") — the modern FastAPI pattern.
# Runs create_tables() once when the server starts.
# Tables already exist? Postgres skips silently. Safe every time.
# ─────────────────────────────────────────────
# Global workflow — set once during lifespan startup
workflow = None
@asynccontextmanager
async def lifespan(app: FastAPI):
    global workflow

    create_tables()  # Your tables - not langgraph
    # Phase 3: LangGraph's tables + compiled async graph
    # This is the only place create_workflow() is called.
    workflow = await create_workflow()

    yield    # server runs here

app = FastAPI(lifespan=lifespan)

class ChatRequest(BaseModel):

    user_input: str
    chat_id: Optional[str] = None  # None = start new chat



@app.get("/health")
async def health_status():
    return {"status": "successfull"}


@app.post("/user")
async def get_user_input(user_request: ChatRequest):
    """
    Request flow with Phase 2 + 3:

    1. Create/reuse chat session                        (Persistence)
    2. Save user message to YOUR messages table         (Persistence)
    3. Build initial State with chat_id                 (State)
    4. ainvoke with thread_id = chat_id                 (Checkpointer)
       ├── orchestrator runs:
       │   ├── loads last 6 messages from DB            (Memory - Phase 2)
       │   ├── injects into LLM prompt
       │   └── saves State snapshot to checkpoints table (Checkpointer)
       └── planner runs:
           ├── loads last 10 messages from DB           (Memory - Phase 2)
           ├── injects into LLM prompt
           └── saves State snapshot to checkpoints table (Checkpointer)
    5. Save AI response to YOUR messages table          (Persistence)
    6. Return chat_id + final_answer
    """
    # ── Step 1: Session management ──────────────────────────
    # New chat if no chat_id provided, else reuse existing session
    chat_id = user_request.chat_id or create_chat()

    # ── Step 2: Persist user message ────────────────────────
    # Save BEFORE calling workflow — so even if workflow crashes,
    # the user's message is already recorded for debugging.
    save_message(chat_id, "user", user_request.user_input)

    # ── Step 3: Build initial State ─────────────────────────
    # chat_id travels through State so agents can access it in Phase 2


    user_data = user_request.user_input
    # LangGraph expects {"key_name": value}
    initial_state = {"user_input": user_data,
                     "chat_id": chat_id,}
    
    # Phase 3: thread_id ties this invoke to LangGraph's checkpoint history
    # Every node execution under this thread_id gets a snapshot in PostgreSQL.
    # If the server crashes mid-graph, the next request with same thread_id
    # can resume from the last successful node — not restart from scratch.
    config = {"configurable": {"thread_id": chat_id}}

    try:
        # ainvoke = async version of invoke
        # Functionally identical — just non-blocking for FastAPI
        response = await workflow.ainvoke(initial_state, config = config)
    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"Workflow failed: {str(e)}")
    
    # ── Phase 1: Persist AI response ─────────────────────────

    final_answer = response.get("final_answer","")
    if final_answer and final_answer != "Failed after retries":
        save_message(chat_id, "ai", final_answer)

    # ── Step 6: Return — include chat_id so client uses it next turn ──
    return {
        "chat_id": chat_id,
        "final_answer": final_answer,
    }

# ── Bonus endpoint: see a chat's full history ───────────────
# Useful for debugging and building a chat UI later
@app.get("/chat/{chat_id}/history")
async def get_chat_history(chat_id: str):
    history = load_chat_history(chat_id)
    return {"chat_id": chat_id, "messages": history}