# main.py
#
# Two endpoints:
#   POST /user         — original blocking endpoint (kept for compatibility)
#   POST /user/stream  — NEW streaming endpoint used by streamlit_streaming.py
#
# Streaming protocol (simple text prefixes, no JSON):
#   data: __chat_id__{id}        → send chat_id first so frontend saves it
#   data: __progress__{label}    → update the status spinner label
#   data: __thinking__{text}     → append a line to the thought-process box
#   data: __token__{word}        → stream the final answer word by word
#   data: __done__               → stream is finished, close connection
#
# WHY text prefixes and not JSON?
#   JSON parsing adds overhead and complexity for both sides.
#   Simple string prefixes are enough for 5 event types.
#   The frontend just does: line.startswith("__thinking__") → strip prefix → display.
#   Easy to read, easy to debug.

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from db.database import create_tables, create_chat, save_message, load_chat_history
from contextlib import asynccontextmanager
from agents.orchestrator import create_workflow
from logger import get_logger
import asyncio

# get_logger(__name__) → logger named "main"
# All output goes to stdout → visible in `docker compose logs -f fastapi`
logger = get_logger(__name__)

# ─────────────────────────────────────────────
# LIFESPAN — runs once at startup, once at shutdown
# create_tables()    → YOUR chats/messages tables
# create_workflow()  → LangGraph tables + compiled async graph with checkpointer
# ─────────────────────────────────────────────
workflow = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global workflow
    logger.info("Starting up — creating DB tables...")
    create_tables()
    logger.info("DB tables ready — initializing LangGraph workflow...")
    workflow = await create_workflow()
    logger.info("Workflow ready — FastAPI is live")
    yield
    logger.info("Shutting down FastAPI")


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    user_input: str
    chat_id: Optional[str] = None   # None = start a new chat session


# ─────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────
@app.get("/health")
async def health_status():
    return {"status": "ok"}          # fixed typo: was "successfull"


# ─────────────────────────────────────────────
# BLOCKING ENDPOINT — original, kept for compatibility
# Use this when you need a simple JSON response (tests, curl, etc.)
# ─────────────────────────────────────────────
@app.post("/user")
async def get_user_input(user_request: ChatRequest):
    chat_id = user_request.chat_id or create_chat()
    logger.info("POST /user | chat_id=%s | input=%s", chat_id, user_request.user_input[:80])
    save_message(chat_id, "user", user_request.user_input)

    initial_state = {"user_input": user_request.user_input, "chat_id": chat_id}
    config = {"configurable": {"thread_id": chat_id}}

    try:
        response = await workflow.ainvoke(initial_state, config=config)
    except Exception as e:
        logger.error("Workflow failed | chat_id=%s | error=%s", chat_id, str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Workflow failed: {str(e)}")

    final_answer = response.get("final_answer", "")
    if final_answer and final_answer != "Failed after retries":
        save_message(chat_id, "ai", final_answer)
        logger.info("POST /user done | chat_id=%s | answer_len=%d", chat_id, len(final_answer))

    return {"chat_id": chat_id, "final_answer": final_answer}


# ─────────────────────────────────────────────
# STREAMING ENDPOINT — used by streamlit_streaming.py
#
# HOW IT WORKS:
#   1. Create/reuse chat session, save user message
#   2. Start astream_events() — this runs the graph AND emits events as each
#      step happens (node starts, tool calls, tool results, node ends)
#   3. For each event we receive, yield an SSE line to the frontend
#   4. After graph finishes, stream the final answer word by word
#   5. Send __done__ to tell frontend the stream is finished
#
# WHY astream_events() and not astream()?
#   astream()        → yields full State snapshots after each node completes
#   astream_events() → yields fine-grained events (tool calls, LLM tokens, node starts)
#   astream_events() gives us the "Searching the web..." granularity we want.
# ─────────────────────────────────────────────
@app.post("/user/stream")
async def stream_user_input(user_request: ChatRequest):

    chat_id = user_request.chat_id or create_chat()
    save_message(chat_id, "user", user_request.user_input)

    initial_state = {"user_input": user_request.user_input, "chat_id": chat_id}
    config = {"configurable": {"thread_id": chat_id}}

    async def event_generator():
        # ── 1. Send chat_id immediately ─────────────────────────────────────
        # Frontend saves this and sends it back next turn for memory
        yield f"data: __chat_id__{chat_id}\n\n"
        logger.info("POST /user/stream started | chat_id=%s | input=%s", chat_id, user_request.user_input[:80])

        final_answer = ""

        try:
            # ── 2. Run the graph with event streaming ───────────────────────
            # version="v2" = latest astream_events protocol (recommended)
            async for event in workflow.astream_events(initial_state, config=config, version="v2"):

                kind = event["event"]       # event type string
                name = event.get("name", "") # node name / tool name / model name
                data = event.get("data", {}) # event payload

                # ── Node started ─────────────────────────────────────────────
                # Fires when each LangGraph node begins executing
                if kind == "on_chain_start":
                    if name == "orchestrator":
                        yield "data: __progress__Orchestrator is routing your request...\n\n"
                    elif name == "researcher":
                        yield "data: __progress__Researcher is gathering real-time data...\n\n"
                    elif name == "planner":
                        yield "data: __progress__Planner is building your itinerary...\n\n"

                # ── Orchestrator node finished ────────────────────────────────
                # Emit the orchestrator's thought + plan into the thinking box.
                # WHY HERE and not on_chain_start?
                #   The thought and plan only exist AFTER the orchestrator's LLM call completes.
                #   on_chain_start fires before the LLM runs — data is still empty.
                #   on_chain_end fires after — output contains current_thought, plan, next_agent.
                #
                # WHY IMPORTANT?
                #   When orchestrator routes directly to planner (no researcher),
                #   no tool calls happen → no on_tool_start/on_tool_end events →
                #   the thinking box is COMPLETELY EMPTY.
                #   This block ensures the thought process is always visible.
                elif kind == "on_chain_end" and name == "orchestrator":
                    output = data.get("output", {})
                    if isinstance(output, dict):
                        thought     = output.get("current_thought", "")
                        plan_items  = output.get("plan", [])
                        next_agent  = output.get("next_agent", "")
                        if thought:
                            yield f"data: __thinking__💭 Thought: {thought}\n\n"
                        if plan_items:
                            plan_summary = " | ".join(plan_items[:4])  # show first 4 plan items
                            if len(plan_items) > 4:
                                plan_summary += f" ... (+{len(plan_items)-4} more)"
                            yield f"data: __thinking__📋 Plan: {plan_summary}\n\n"
                        if next_agent:
                            arrow = {"researcher": "🔎 researcher", "planner": "📝 planner", "end": "🔚 end"}
                            yield f"data: __thinking__➡️ Routing → {arrow.get(next_agent, next_agent)}\n\n"

                # ── Tool call started ─────────────────────────────────────────
                # Fires when Tavily (or any tool) is called inside the researcher
                # name = tool function name, data["input"] = the arguments passed
                elif kind == "on_tool_start":
                    tool_input = data.get("input", {})
                    # tool_input is a dict with the tool's parameters
                    if isinstance(tool_input, dict):
                        query = tool_input.get("query", str(tool_input))
                    else:
                        query = str(tool_input)
                    # Truncate long queries so they fit cleanly in the UI
                    query_display = (query[:80] + "...") if len(query) > 80 else query
                    yield f"data: __thinking__🔍 Searching: {query_display}\n\n"

                # ── Tool call finished ────────────────────────────────────────
                # Fires after the tool returns its result
                elif kind == "on_tool_end":
                    yield f"data: __thinking__📋 Got search results\n\n"

                # ── Planner node finished ─────────────────────────────────────
                # The planner is ALWAYS the last node — its output = final_answer
                # We capture it here and stream it word-by-word below
                elif kind == "on_chain_end" and name == "planner":
                    output = data.get("output", {})
                    if isinstance(output, dict):
                        final_answer = output.get("final_answer", "")

        except Exception as e:
            logger.error("Stream error | chat_id=%s | error=%s", chat_id, str(e), exc_info=True)
            yield f"data: __thinking__❌ Error: {str(e)}\n\n"
            yield "data: __done__\n\n"
            return

        # ── 3. Persist AI response ──────────────────────────────────────────
        if final_answer and final_answer != "Failed after retries":
            save_message(chat_id, "ai", final_answer)
            logger.info("Stream complete | chat_id=%s | answer_len=%d", chat_id, len(final_answer))

        # ── 4. Stream answer word by word ───────────────────────────────────
        # WHY word-by-word and not all at once?
        #   Sending the entire answer as one __token__ event works, but streaming
        #   word by word creates the "typing" effect the user sees in ChatGPT.
        #   0.02s delay = ~50 words/sec — fast enough to feel snappy, slow enough
        #   to see the text appearing.
        answer_to_stream = final_answer or "Sorry, I could not generate an answer. Please try again."
        words = answer_to_stream.split(" ")
        for i, word in enumerate(words):
            # Add space after each word except the last
            chunk = word + (" " if i < len(words) - 1 else "")
            yield f"data: __token__{chunk}\n\n"
            await asyncio.sleep(0.02)   # 20ms delay = visible typing effect

        # ── 5. Signal completion ────────────────────────────────────────────
        yield "data: __done__\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",    # tells Nginx NOT to buffer — required for SSE
            "Connection": "keep-alive",
        }
    )


# ─────────────────────────────────────────────
# HISTORY ENDPOINT — debug/inspect any chat
# ─────────────────────────────────────────────
@app.get("/chat/{chat_id}/history")
async def get_chat_history(chat_id: str):
    history = load_chat_history(chat_id)
    return {"chat_id": chat_id, "messages": history}
