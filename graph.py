from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo")
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
import textwrap

# Import tool utilities
from langchain_experimental.tools.python.tool import PythonREPLTool
from langgraph.prebuilt import ToolNode

# Initialize the Python REPL tool
python_repl_tool = PythonREPLTool()
tools = [python_repl_tool]
llm_with_tools = llm.bind_tools(tools)


def assistant(state: MessagesState):
    sys_msg = SystemMessage(content="""You are personal assistant for learning to code. 
    You can execute Python code to help demonstrate concepts or test code snippets.
    
    IMPORTANT: When you use the Python REPL tool to execute code, you MUST:
    1. Tell the user you're going to run code.
    2. Show the code you're running (in a code block if possible).
    3. After getting the result, explain what happened.
    4. **Finally, if you used the Python REPL tool, conclude your response with the exact phrase: "Python Tool Used 🐍"**
    
    Example responses:
    - "Let me calculate that for you using Python... [code] ... The result is... Python Tool Used 🐍"
    - "I'll run this code to demonstrate... [code] ... Here's what happened. Python Tool Used 🐍"
    
    This helps users learn when and how code execution works!
    """
)
    response = llm_with_tools.invoke([sys_msg] + state['messages'])
    return {'messages': response}

def should_continue(state: MessagesState):
    messages = state['messages']
    last_message = messages[-1]
    # If there are tool calls, route to the tool node
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    # Otherwise, end the conversation
    return END
tool_node = ToolNode(tools)

from langgraph.graph import START, StateGraph, END

from langgraph.checkpoint.redis import RedisSaver
import os
import redis
# Get Redis URL from environment or use default
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
try:
    memory = RedisSaver(redis_url)
    print(f"✅ RedisSaver initialized with URL: {redis_url.split('@')[-1]}") # Log URL without password
except Exception as e:
    print(f"❌ CRITICAL ERROR: Failed to initialize RedisSaver: {e}")
    # If Redis is critical for your app, you might want to exit here too,
    # or fall back to MemorySaver for local dev. For deployment, usually critical.
    import sys
    sys.exit(1)


builder = StateGraph(MessagesState)
builder.add_node('chat', assistant)
builder.add_node('tools', tool_node)  # THIS IS THE TOOL NODE
builder.add_edge(START, 'chat')
builder.add_conditional_edges('chat', should_continue, ['tools', END])
builder.add_edge('tools', 'chat')
graph = builder.compile(checkpointer=memory)


# --- FastAPI integration for deployment ---
from fastapi import FastAPI, Request
from langchain_core.messages import HumanMessage

app = FastAPI()

@app.post("/chat")
def chat(request: dict):
    print("Received:", request)
    user_input = request.get("message")
    thread_id = request.get("session_id") or request.get("thread_id") or "default-thread"  # Accept both
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config)
    response = result["messages"][-1].content
    return {"response": response}


