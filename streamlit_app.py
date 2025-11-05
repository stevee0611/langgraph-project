import os
import json
import requests
import streamlit as st
import uuid
import time
from typing import Any, Dict, List

BACKEND_URL = os.environ.get(
    "BACKEND_URL",
    "https://langgraph-project-production.up.railway.app/chat",
)
REQUEST_TIMEOUT = 30

st.set_page_config(page_title="LangGraph Chat", page_icon="🤖")
st.title("🤖 Personal Coding Assistant")
st.write("Ask your AI assistant anything about coding!")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = []

col1, col2 = st.columns([1, 9])
with col1:
    if st.button("Clear"):
        st.session_state.messages = []
with col2:
    st.markdown("")

def extract_reply_from_backend(data: Any) -> str:
    try:
        if isinstance(data, dict):
            if "response" in data and isinstance(data["response"], str):
                return data["response"]

            if "messages" in data and isinstance(data["messages"], list) and data["messages"]:
                last = data["messages"][-1]
                if isinstance(last, dict):
                    for k in ("content", "text", "message"):
                        if k in last and isinstance(last[k], str):
                            return last[k]
                    return str(last)
                else:
                    return str(last)

            if "message" in data and isinstance(data["message"], str):
                return data["message"]

            if "error" in data:
                return f"Server error: {data['error']}"

        return str(data)
    except Exception as e:
        return f"Failed to parse response: {e}"

def send_message_to_backend(user_input: str) -> str:
    payload = {
        "message": user_input,
        "session_id": st.session_state.session_id,
    }

    try:
        with requests.post(BACKEND_URL, json=payload, stream=True, timeout=REQUEST_TIMEOUT) as r:
            r.raise_for_status()

            message_placeholder = st.empty()
            full_response = ""

            # Show thinking indicator
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown("*Thinking...*")

            for line in r.iter_lines():
                if line:
                    try:
                        json_response = json.loads(line)
                        chunk = json_response.get('content', '')
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                        time.sleep(0.01)  # Add small delay for more natural typing effect
                    except json.JSONDecodeError:
                        continue

            # Remove thinking indicator
            thinking_placeholder.empty()
            message_placeholder.markdown(full_response)
            return full_response

    except requests.exceptions.RequestException as e:
        return f"Network error: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"

def send_message(user_input: str):
    # Immediately display user message
    user_message = st.chat_message("user")
    user_message.write(user_input)

    # Add to message history
    st.session_state.messages.append({"role": "user", "message": user_input})

    # Create assistant message container before API call
    assistant_message = st.chat_message("assistant")
    with assistant_message:
        ai_reply = send_message_to_backend(user_input)
        st.session_state.messages.append({"role": "assistant", "message": ai_reply})

# Display message history first
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["message"])

# Move chat input to the end
user_input = st.chat_input("Type your message here...")
if user_input:
    send_message(user_input)
