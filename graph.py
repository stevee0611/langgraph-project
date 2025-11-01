from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from typing import List
from langchain_core.documents import Document
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
import textwrap
from langchain_tavily import TavilySearch

class GraphState(MessagesState):
    """
    Represents the state of our graph.

    Attributes:
        messages: The history of messages.
        documents: A list of documents retrieved from the vector store.
    """
    documents: List[Document]

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
from langchain_community.vectorstores import Qdrant

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
python_repl_tool = PythonREPLTool()
tools = [python_repl_tool]
llm_with_tools = llm.bind_tools(tools)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal

class RouteQuery(BaseModel):
    """Route a user query to the appropriate tool."""
    datasource: Literal["vectorstore", "chat"] = Field(
        ...,
        description="Given a user query, route it to `vectorstore` if it requires searching for specific documents, or to `chat` for all other cases.",
    )
# Create a prompt template and bind it to the LLM with our desired output structure.
structured_llm = llm.with_structured_output(RouteQuery)
router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are an expert at routing a user query to a vectorstore or to a general chat. Use the vectorstore for questions that require fetching specific information from documents."),
        ("human", "{question}"),
    ]
)
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
        print("---ROUTING TO RETRIEVE---")
        return "retrieve"
    else:
        print("---ROUTING TO CHAT---")
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


def assistant(state: GraphState):
    """
    Main assistant logic. Uses retrieved PDF documents, web results, and general LLM knowledge.
    Clearly tells the user which source the answer comes from.
    """
    print("---CALLING LLM---")

    # Prepare context from retrieved documents
    contexts = []

    if state.get("documents"):
        pdf_context = "\n---\n".join([doc.page_content for doc in state["documents"]])
        contexts.append(f"[Documents 📄]\n{pdf_context}")

    if state.get("web_documents"):
        web_context = "\n---\n".join([doc.page_content for doc in state["web_documents"]])
        contexts.append(f"[Web 🌐]\n{web_context}")

    combined_context = "\n\n".join(contexts) if contexts else None

    # --- System message construction ---
    if combined_context:
        # Keep the original RAG prompt logic but add web context and source instruction
        sys_msg_content = textwrap.dedent(f"""You are a personal assistant for learning to code.
        You have access to documents and/or web search results relevant to the user's question. 
        When you use information from these sources to answer, you MUST explicitly tell the user which source it came from by using:
            - "Information retrieved from documents 📄" for PDF content
            - "Information retrieved from the web 🌐" for web content
        If you answer from your general knowledge without using retrieved content, state: "General response based on knowledge."

        Use the following context to answer the user's question. If the context does not have the answer, say you don't know.

        CONTEXT:
        {combined_context}

        ---
        You can also execute Python code to help demonstrate concepts or test code snippets.

        IMPORTANT: When you use the Python REPL tool to execute code, you MUST:
        1. Tell the user you're going to run code.
        2. Show the code you're running (in a code block if possible).
        3. After getting the result, explain what happened.
        4. **Finally, if you used the Python REPL tool, conclude your response with the exact phrase: "Python Tool Used 🐍"**
        """)
    else:
        # Keep the original general-purpose system message
        sys_msg_content = textwrap.dedent("""You are a personal assistant for learning to code. 
        You can execute Python code to help demonstrate concepts or test code snippets.

        IMPORTANT: When you use the Python REPL tool to execute code, you MUST:
        1. Tell the user you're going to run code.
        2. Show the code you're running (in a code block if possible).
        3. After getting the result, explain what happened.
        4. **Finally, if you used the Python REPL tool, conclude your response with the exact phrase: "Python Tool Used 🐍"**
        """)

    sys_msg = SystemMessage(content=sys_msg_content)

    response = llm_with_tools.invoke([sys_msg] + state['messages'])
    return {'messages': [response]}


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
builder.add_node('web_retrieve', web_retrieve) # Web retrieval (new)
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


