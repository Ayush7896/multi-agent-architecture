# """
# streamlit_app.py

# WHAT: Chat UI in the browser. Talks to FastAPI backend.
# WHY:  Users need a visual interface — not curl commands.
# HOW:  Streamlit = Python library that builds web UIs with no HTML/CSS/JS.
#       st.chat_message() = the chat bubble UI.
#       st.session_state = stores chat_id and history across rerenders.
#       requests.post() = HTTP call to FastAPI — Streamlit is just a client.

# WHY Streamlit and not React?
#   For internal tools and demos: Streamlit = much faster to build.
#   For production consumer apps: React/Next.js = better UX control.
#   At Deloitte/Mercedes: Streamlit POCs → React production. You start here.

# Run: streamlit run streamlit_app.py
# """

import streamlit as st
import requests
import os

# FastAPI URL — from env var (changes between local / docker / k8s)
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Travel Planner AI",
    page_icon="✈️",
    layout="centered"
)
st.title("✈️ Travel Planner AI")
st.caption("Powered by real-time research + multi-agent AI")

# ── Session state — persists across Streamlit rerenders ──────────────────────
# WHY session_state?
#   Streamlit reruns the ENTIRE script on every user interaction.
#   Without session_state, chat history disappears on every message.
#   session_state survives rerenders within the same browser session.

if "chat_id" not in st.session_state:
    st.session_state.chat_id = None      # None = new chat on first message

if "messages" not in st.session_state:
    st.session_state.messages = []         # list of {role, content}

# ── Display existing messages ─────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Plan a trip... e.g. 'Plan a 3-day Paris Trip'"):

    # show user messages immediately
    st.session_state.messages.append({'role':"user","content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # call fastapi - with loading spinner
    with st.chat_message("assistant"):
        with st.spinner("Researching and planning... (this takes 15-30 seconds)"):
            try:
                response = requests.post(
                    f"{FASTAPI_URL}/user",
                    json = {
                        "user_input": user_input,
                        "chat_id": st.session_state.chat_id   # None on first turn
                    },
                    timeout = 120 # research can take time
                )
                response.raise_for_status()
                data = response.json()

                # save chat_id for next turn (multi-turn memory)
                st.session_state.chat_id = data["chat_id"]

                answer = data.get("final_answer", "Sorry,I could not generate the answer")
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except requests.exceptions.Timeout:
                st.error("Request timed out. Research is taking too long — try a simpler query.")
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot connect to FastAPI at {FASTAPI_URL}. Is it running?")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ── Sidebar — session info and controls ──────────────────────────────────────
with st.sidebar:
    st.header("Session")
    if st.session_state.chat_id:
        st.code(st.session_state.chat_id[:8] + "...", language=None)
        st.caption("Chat ID (sent with each message for memory)")

    if st.button("New conversation"):
        st.session_state.chat_id = None
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("FastAPI: " + FASTAPI_URL)