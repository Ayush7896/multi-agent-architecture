"""
Streamlit streaming — kept simple and easy to read.

Key concept:
  requests.post(stream=True) = keeps HTTP connection open
  response.iter_lines()      = reads one line at a time as they arrive
  Each line = one SSE event  = one token from FastAPI

  We use simple string prefixes (__token__, __thinking__, __done__)
  instead of JSON — easier to parse, easier to understand.
"""

import streamlit as st
import requests
import os

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

st.set_page_config(page_title="Travel Planner AI", page_icon="✈️")
st.title("✈️ Travel Planner AI")

# Session state — survives Streamlit rerenders
if "chat_id"  not in st.session_state: st.session_state.chat_id  = None
if "messages" not in st.session_state: st.session_state.messages = []

# Show previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Show thinking if it exists (collapsed by default)
        if msg.get("thinking"):
            with st.expander("💭 Thought process", expanded=False):
                st.write(msg["thinking"])
        st.markdown(msg["content"])

# Chat input box at the bottom
if user_input := st.chat_input("Where do you want to travel?"):

    # 1. Show the user's message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 2. Show AI response — this is where streaming happens
    # streamlit_app.py — only the inside of the streaming block changes

    with st.chat_message("assistant"):
        thinking_box  = st.status("🧠 Thinking...", expanded=True)
        thinking_text = ""
        answer_box    = st.empty()
        answer_text   = ""

        try:
            with requests.post(
                f"{FASTAPI_URL}/user/stream",
                json={"user_input": user_input, "chat_id": st.session_state.chat_id},
                stream=True,
                timeout=120
            ) as resp:

                for line in resp.iter_lines():
                    line = line.decode("utf-8") if isinstance(line, bytes) else line
                    if not line.startswith("data: "):
                        continue

                    payload = line[6:]

                    if payload.startswith("__chat_id__"):
                        st.session_state.chat_id = payload.replace("__chat_id__", "")

                    elif payload.startswith("__thinking__"):
                        thinking_text += payload.replace("__thinking__", "")
                        with thinking_box:
                            st.write(thinking_text)

                    elif payload.startswith("__progress__"):
                        label = payload.replace("__progress__", "")
                        thinking_box.update(label=f"⏳ {label}")

                    elif payload.startswith("__token__"):
                        token = payload.replace("__token__", "")
                        answer_text += token

                        # ── KEY FIX ───────────────────────────────────────────
                        # During streaming: plain text only — no markdown parsing
                        # st.text() renders exactly what you give it, no interpretation
                        # User still sees words appearing token by token ✓
                        # Markdown symbols like ##, **, - show as plain characters
                        # but that is fine — it looks clean and streams smoothly
                        answer_box.text(answer_text + " ▌")

                    elif payload == "__done__":
                        # ── Stream finished: NOW render full markdown ──────────
                        # At this point answer_text is complete and well-formed
                        # st.markdown() renders ## as headers, ** as bold, - as bullets
                        # This is the "reveal" moment — plain text → beautiful markdown
                        answer_box.markdown(answer_text)

                        thinking_box.update(
                            label="💭 Thought process",
                            state="complete",
                            expanded=False
                        )
                        break

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to FastAPI. Is it running?")
        except Exception as e:
            st.error(f"Error: {e}")

        st.session_state.messages.append({
            "role":     "assistant",
            "content":  answer_text,
            "thinking": thinking_text
        })

# Sidebar
with st.sidebar:
    st.header("Session")
    if st.session_state.chat_id:
        st.code(st.session_state.chat_id[:8] + "...", language=None)
        st.caption("Send this back for memory across turns")
    if st.button("🗑️ New chat"):
        st.session_state.chat_id = None
        st.session_state.messages = []
        st.rerun()