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

conversation = [HumanMessage('What is my name and what do you recommend me to learn today?')]
conversation = graph.invoke({"messages": conversation}, config)
for m in conversation["messages"]:
    print(f"=== {m.type.capitalize()} Message ===")
    for line in textwrap.wrap(m.content, width=80):
        print(line)
    print()

    from fastapi import FastAPI

    app = FastAPI()