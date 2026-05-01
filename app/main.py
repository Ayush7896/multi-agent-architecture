# Persistence lives HERE — in the FastAPI layer, outside LangGraph.
# LangGraph knows nothing about your chats/messages tables.

# Request lifecycle:
#   1. Receive request
#   2. Create or reuse chat session
#   3. Save user message  ← Persistence
#   4. Run workflow       ← LangGraph (State lives here)
#   5. Save AI response   ← Persistence
#   6. Return response

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agents.orchestrator import workflow
from typing import Optional
from db.database import create_tables, create_chat,save_message, load_chat_history
from contextlib import asynccontextmanager



# ─────────────────────────────────────────────
# LIFESPAN
# Replaces @app.on_event("startup") — the modern FastAPI pattern.
# Runs create_tables() once when the server starts.
# Tables already exist? Postgres skips silently. Safe every time.
# ─────────────────────────────────────────────

async def lifespan(app: FastAPI):
    create_tables()  # Your tables - not langgraph
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
    How chat_id works across turns:

    Turn 1:  client sends  { "user_input": "Plan Paris trip" }
             server creates new chat_id = "abc-123"
             server returns { "chat_id": "abc-123", "final_answer": "..." }

    Turn 2:  client sends  { "user_input": "Add Rome", "chat_id": "abc-123" }
             server reuses existing chat — messages table already has Turn 1
             In Phase 2, agents will load Turn 1 messages as Memory

    This is how multi-turn conversation works.
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
    # ── Step 4: Run LangGraph workflow ──────────────────────
    # State is temporary — lives only during this invoke() call.
    # Persistence is what makes the conversation permanent.
    try:
        response = workflow.invoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code = 500, detaile = f"Workflow failed: {str(e)}")
    
    # ── Step 5: Persist AI response ─────────────────────────

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