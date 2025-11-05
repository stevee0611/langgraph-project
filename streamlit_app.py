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
        response = requests.post(BACKEND_URL, json=payload, stream=True, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        full_response = ""
        got_chunk = False

        # Try to read streaming JSON-lines first
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    json_response = json.loads(line)
                    # Extract the actual content from the response
                    chunk = extract_reply_from_backend(json_response)
                    if chunk:
                        full_response += chunk
                        got_chunk = True
                        # Yield the partial response
                        yield full_response
                except json.JSONDecodeError:
                    # Not a JSON line — treat as plain text
                    text_line = line if isinstance(line, str) else line.decode('utf-8', errors='ignore')
                    if text_line.strip():
                        full_response += text_line
                        got_chunk = True
                        yield full_response

        # If nothing was streamed, attempt to parse the full response as JSON (non-streaming fallback)
        if not got_chunk:
            try:
                json_response = response.json()
                chunk = extract_reply_from_backend(json_response)
                if chunk:
                    full_response += chunk
                    yield full_response
                else:
                    # Fallback to raw text
                    text = response.text.strip()
                    if text:
                        full_response += text
                        yield full_response
            except ValueError:
                # response is not JSON, use text
                text = response.text.strip()
                if text:
                    full_response += text
                    yield full_response

        return full_response if full_response else "No response received"

    except requests.exceptions.RequestException as e:
        return f"Network error: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"

def send_message(user_input: str):
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)

    # Add to message history
    st.session_state.messages.append({"role": "user", "message": user_input})

    # Display assistant message
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        # Show initial thinking indicator
        message_placeholder.markdown("*Thinking...*")

        partial_response = ""
        # Get streaming response
        for chunk in send_message_to_backend(user_input):
            partial_response = chunk
            # Ensure emojis and formatting are preserved
            formatted_response = partial_response.replace("```", "\n```\n")  # Fix code block formatting
            message_placeholder.markdown(formatted_response + "▌", unsafe_allow_html=True)
            time.sleep(0.01)

        # Show final response
        final_response = partial_response if partial_response else "No response received"
        # Ensure final message preserves formatting
        message_placeholder.markdown(final_response, unsafe_allow_html=True)

        # Add to message history
        st.session_state.messages.append({"role": "assistant", "message": final_response})

# Display message history first
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["message"])

# Move chat input to the end
user_input = st.chat_input("Type your message here...")
if user_input:
    send_message(user_input)
