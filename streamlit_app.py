# app.py
import os
import requests
import streamlit as st
from typing import Any, Dict, List

# -------- CONFIG --------
# Default points to your Railway deployment /chat endpoint.
# If you want to override locally, set env var BACKEND_URL.
BACKEND_URL = os.environ.get(
    "BACKEND_URL",
    "https://langgraph-project-production.up.railway.app/chat",
)
REQUEST_TIMEOUT = 30  # seconds

# --------- UI SETUP ---------
st.set_page_config(page_title="LangGraph Chat", page_icon="🤖")
st.title("🤖 LangGraph Chat Assistant")
st.write("Ask your AI assistant anything — responses are plain text.")

# --------- Session State ---------
if "messages" not in st.session_state:
    # list of {"role": "user"|"assistant", "message": str}
    st.session_state.messages: List[Dict[str, str]] = []

# Optional quick clear button
col1, col2 = st.columns([1, 9])
with col1:
    if st.button("Clear"):
        st.session_state.messages = []
with col2:
    st.markdown("")

# --------- Helpers ---------
def extract_reply_from_backend(data: Any) -> str:
    """
    Try to extract a human-readable reply from several common backend shapes:
    - {"response": "text"}
    - {"messages": [{"content": "..."}, ...]}
    - {"message": "text"}
    Fallback: stringify the whole response.
    """
    try:
        if isinstance(data, dict):
            # explicit single-response API
            if "response" in data and isinstance(data["response"], str):
                return data["response"]

            # sometimes server returns messages list
            if "messages" in data and isinstance(data["messages"], list) and data["messages"]:
                last = data["messages"][-1]
                if isinstance(last, dict):
                    # try common keys
                    for k in ("content", "text", "message"):
                        if k in last and isinstance(last[k], str):
                            return last[k]
                    # fallback to str(last)
                    return str(last)
                else:
                    return str(last)

            # server may echo 'message'
            if "message" in data and isinstance(data["message"], str):
                return data["message"]

            # any error field
            if "error" in data:
                return f"Server error: {data['error']}"

        # fallback: return stringified JSON/text
        return str(data)
    except Exception as e:
        return f"Failed to parse response: {e}"

def send_message_to_backend(user_input: str) -> str:
    """
    POST to BACKEND_URL with {"message": user_input}.
    Returns the assistant reply (plain string) or an error message.
    """
    payload = {"message": user_input}
    try:
        resp = requests.post(BACKEND_URL, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return extract_reply_from_backend(data)
    except requests.exceptions.RequestException as e:
        # network / HTTP error
        return f"Network error: {e}"
    except ValueError:
        # JSON decoding error
        return "Invalid JSON response from server."
    except Exception as e:
        return f"Unexpected error: {e}"

# --------- Send / Render Logic ---------
def send_message(user_input: str):
    # save user message immediately
    st.session_state.messages.append({"role": "user", "message": user_input})

    # call backend with spinner
    with st.spinner("Thinking..."):
        ai_reply = send_message_to_backend(user_input)

    # append assistant reply
    st.session_state.messages.append({"role": "assistant", "message": ai_reply})

# Chat input (top-level so Streamlit can capture Enter)
user_input = st.chat_input("Type your message here...")

if user_input:
    send_message(user_input)

# Display chat history in order
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["message"])
    else:
        st.chat_message("assistant").write(msg["message"])

