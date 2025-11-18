from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import List
from langchain_core.documents import Document
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
import textwrap
from langchain_tavily import TavilySearch
from langchain.agents import Tool

load_dotenv()

class GraphState(MessagesState):
    """
    Represents the state of our graph.

    Attributes:
        messages: The history of messages.
        documents: A list of documents retrieved from the vector store.
    """
    documents: List[Document]
    web_documents: List[Document]
    session_id: str

try:
    from langchain.callbacks.manager import CallbackManager
    print("✅ SUCCESS: langchain.callbacks was imported successfully.")
except ImportError as e:
    print(f"❌ CRITICAL FAILURE: Could not import from langchain.callbacks. Error: {e}")
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Import tool utilities
from langchain_experimental.tools.python.tool import PythonREPLTool
from langgraph.prebuilt import ToolNode
import os
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import Qdrant

embeddings = OpenAIEmbeddings()
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)
vector_store = Qdrant(
    client=qdrant_client,
    collection_name="my_docs",
    embeddings=embeddings,
)
retriever = vector_store.as_retriever()


# Initialize the Python REPL tool


from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal

class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "web_search", "chat"] = Field(
        ...,
        description="Route to `vectorstore` for PDF questions, `web_search` for current info, or `chat` for general conversation.",
    )
# Create a prompt template and bind it to the LLM with our desired output structure.
structured_llm = llm.with_structured_output(RouteQuery)
router_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Route queries to:\n"
     "- `vectorstore` for questions about uploaded documents\n"
     "- `web_search` for current events, latest info, or general knowledge\n"
     "- `chat` for greetings or simple conversations"
    ),
    ("human", "{question}"),
])

question_router = router_prompt | structured_llm


def route_question(state: GraphState):
    """
    Routes the user's question to determine if we need to retrieve documents or not.
    """
    print("---ROUTING QUESTION---")
    question = state["messages"][-1].content

    # Call the router
    result = question_router.invoke({"question": question})

    if result.datasource == 'vectorstore':
        return "retrieve"
    elif result.datasource == 'web_search':
        return "web_retrieve"  # Now this can be triggered!
    else:
        return "chat"

def retrieve(state: GraphState):
    """
    Retrieves documents from the vector store.

    Args:
        state (GraphState): The current graph state.

    Returns:
        GraphState: New state with retrieved documents.
    """
    print("---RETRIEVING DOCUMENTS---")
    # Get the most recent question
    question = state["messages"][-1].content

    # Get session_id from state (will be passed from FastAPI)
    session_id = state.get("session_id")

    # Try to retrieve session-specific documents first
    if session_id:
        session_docs = retriever.invoke(
            question,
            filter={"session_id": session_id}
        )
        if session_docs:
            print(f"---FOUND {len(session_docs)} SESSION DOCS---")
            return {"documents": session_docs}

    # Retrieve documents
    docs = retriever.invoke(question)
    print("---DOCUMENTS RETRIEVED---")
    # Add them to the state
    return {"documents": docs}

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")  # type: str
tavily_tool = TavilySearch(
    api_key=TAVILY_API_KEY,
    max_results=5,
    topic="general"
)

def web_retrieve(query: str):
    results = tavily_tool.invoke({"query": query})
    # Now wrap results as Documents (your import)
    return [Document(page_content=item["content"], metadata={"url": item["url"]}) for item in results["results"]]

web_tool = Tool(
    name="Web_Search",
    func=web_retrieve,  # your existing function returning Documents
    description=(
        "Use this to search the web for questions not answered by uploaded documents. "
        "Return relevant information in text form."
    )
)

python_repl_tool = PythonREPLTool()
tools = [python_repl_tool, web_tool]
llm_with_tools = llm.bind_tools(tools)

def assistant(state: GraphState):
    contexts = []

    if state.get("documents"):
        pdf_context = "\n---\n".join([doc.page_content for doc in state["documents"]])
        contexts.append(f"[Documents 📄]\n{pdf_context}")

    if state.get("web_documents"):
        web_context = "\n---\n".join([doc.page_content for doc in state["web_documents"]])
        contexts.append(f"[Web 🌐]\n{web_context}")

    combined_context = "\n\n".join(contexts) if contexts else None

    if combined_context:
        sys_msg_content = textwrap.dedent(f"""You are a personal assistant for learning to code.
        You have access to documents and/or web search results relevant to the user's question.
        When you use information from these sources to answer, you MUST explicitly tell the user which source it came from:
            - "Information retrieved from documents 📄" for PDF content
            - "Information retrieved from the web 🌐" for web content
            - "General response based on knowledge" if neither is used.

        CONTEXT:
        {combined_context}

        You can also execute Python code to demonstrate or test concepts.

        IMPORTANT: When you use the Python REPL tool to execute code:
        1. Tell the user you're running code.
        2. Show the code (in a code block).
        3. Explain the result.
        4. Conclude with "Python Tool Used 🐍".
        """)
    else:
        # fallback general prompt
        sys_msg_content = textwrap.dedent("""You are a personal assistant for learning to code.
        You can execute Python code to demonstrate or test concepts.
        Follow the Python REPL tool rules if you use it.
        """)

    sys_msg = SystemMessage(content=sys_msg_content)
    response = llm_with_tools.invoke([sys_msg] + state['messages'])
    return {'messages': [response]}


def web_retrieve_node(state: GraphState):
    """
    Retrieves web results and adds them to state.
    """
    print("---WEB SEARCH---")
    question = state["messages"][-1].content

    # Call the Tavily search
    results = tavily_tool.invoke({"query": question})

    # Convert to Documents
    web_docs = [
        Document(page_content=item["content"], metadata={"url": item["url"]})
        for item in results["results"]
    ]

    print(f"---WEB SEARCH COMPLETE: {len(web_docs)} results---")

    # Return with correct state key
    return {"web_documents": web_docs}


def should_continue(state: GraphState):
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
# Get Redis URL from the environment or use default
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
try:
    memory = RedisSaver(redis_url)
    print(f"✅ RedisSaver initialized with URL: {redis_url.split('@')[-1]}") # Log URL without a password
except Exception as e:
    print(f"❌ CRITICAL ERROR: Failed to initialize RedisSaver: {e}")
    # If Redis is critical for your app, you might want to exit here too,
    # or fall back to MemorySaver for local dev. For deployment, usually critical.
    import sys
    sys.exit(1)


builder = StateGraph(GraphState)

# Nodes
builder.add_node('retrieve', retrieve)         # PDF retrieval
builder.add_node('web_retrieve', web_retrieve_node) # Web retrieval (new)
builder.add_node('chat', assistant)            # Assistant with combined context
builder.add_node('tools', tool_node)           # Python REPL or other tools

# Entry point routing
builder.add_conditional_edges(
    START,
    route_question,
    {
        "retrieve": "retrieve",
        "web_retrieve": "web_retrieve",
        "chat": "chat",
    }
)

# Connect retrieval nodes to chat
builder.add_edge('retrieve', 'chat')
builder.add_edge('web_retrieve', 'chat')

# Conditional routing from chat to tools or END
builder.add_conditional_edges('chat', should_continue, ['tools', END])
builder.add_edge('tools', 'chat')

graph = builder.compile(checkpointer=memory)


# --- FastAPI integration for deployment ---
from fastapi import FastAPI, Request
app = FastAPI()

@app.post("/chat")
def chat(request: dict):
    print("Received:", request)
    user_input = request.get("message")
    session_id = request.get("session_id") or request.get("thread_id") or "default-thread"

    config = {"configurable": {"thread_id": session_id}}

    # Pass session_id in the state
    result = graph.invoke({
        "messages": [HumanMessage(content=user_input)],
        "session_id": session_id
    }, config)
    response = result["messages"][-1].content
    return {"response": response}


