from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo")
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
import textwrap

def assistant(state: MessagesState):
    sys_msg = SystemMessage(content="You are Sardor's personal assistant for learning to code")
    response = llm.invoke([sys_msg] + state['messages'])
    return {'messages': response}
from langgraph.graph import START, StateGraph, END

from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
config = {"configurable": {"thread_id": "1"}}

builder = StateGraph(MessagesState)
builder.add_node('chat', assistant)
builder.add_edge(START, 'chat')
builder.add_edge('chat', END)
graph = builder.compile(checkpointer=memory)


# --- FastAPI integration for deployment ---
from fastapi import FastAPI, Request
from langchain_core.messages import HumanMessage

app = FastAPI()

@app.post("/chat")
def chat(request: dict):
    print("Received:", request)
    user_input = request.get("message")
    result = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config)
    response = result["messages"][-1].content
    return {"response": response}


