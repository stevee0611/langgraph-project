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
from qdrant_client.http import models
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import Qdrant, QdrantVectorStore

embeddings = OpenAIEmbeddings()

# Initialize Qdrant client with error handling
qdrant_client = None
retriever = None

try:
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=60,
        prefer_grpc=False,
    )
    print(f"✅ Qdrant client connected to {os.getenv('QDRANT_URL')}")

    # Check if collection exists (DON'T create it!)
    collection_name = "my_docs"
    try:
        collection_info = qdrant_client.get_collection(collection_name)
        print(f"✅ Collection '{collection_name}' exists with {collection_info.points_count} documents")

        # Initialize vector store only if collection exists
        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name="my_docs",
            embedding=embeddings,
        )
        retriever = vector_store.as_retriever()
        print("✅ Vector store initialized successfully")

    except Exception as e:
        print(f"⚠️ Collection '{collection_name}' not found: {e}")
        print(f"   Run 'python loader.py' to create the collection and upload documents.")
        retriever = None

except Exception as e:
    print(f"⚠️ Qdrant connection failed: {e}")
    print(f"   Document retrieval will be disabled. Chat and web search will still work.")
    qdrant_client = None
    retriever = None

# Initialize the Python REPL tool


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

    # Check if retriever is available
    if retriever is None:
        print("⚠️ Retriever not available, skipping document retrieval")
        return {"documents": []}

    # Get the most recent question
    question = state["messages"][-1].content

    # Get session_id from state (will be passed from FastAPI)
    session_id = state.get("session_id")

    try:
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
        print(f"---DOCUMENTS RETRIEVED: {len(docs)}---")
        # Add them to the state
        return {"documents": docs}
    except Exception as e:
        print(f"⚠️ Error retrieving documents: {e}")
        return {"documents": []}

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
    source_used = None  # Track which source was used

    if state.get("documents") and len(state.get("documents", [])) > 0:
        pdf_context = "\n---\n".join([doc.page_content for doc in state["documents"]])
        contexts.append(f"[Documents 📄]\n{pdf_context}")
        source_used = "documents"

    if state.get("web_documents") and len(state.get("web_documents", [])) > 0:
        web_context = "\n---\n".join([doc.page_content for doc in state["web_documents"]])
        contexts.append(f"[Web 🌐]\n{web_context}")
        if source_used == "documents":
            source_used = "both"
        else:
            source_used = "web"

    combined_context = "\n\n".join(contexts) if contexts else None

    if combined_context:
        sys_msg_content = textwrap.dedent(f"""You are a personal assistant for learning to code.
        You have access to documents and/or web search results relevant to the user's question.
        Use the provided context to answer accurately.

        CONTEXT:
        {combined_context}

        You can also execute Python code to demonstrate or test concepts.
        """)
    else:
        # fallback general prompt
        sys_msg_content = textwrap.dedent("""You are a personal assistant for learning to code.
        You can execute Python code to demonstrate or test concepts.
        """)

    sys_msg = SystemMessage(content=sys_msg_content)
    response = llm_with_tools.invoke([sys_msg] + state['messages'])

    # ✅ ALWAYS force add source tag - don't rely on LLM
    if hasattr(response, 'content') and response.content:
        content = response.content

        # Skip tagging if this is a tool call (will be handled after tool execution)
        if not (hasattr(response, 'tool_calls') and response.tool_calls):
            # Check if tag already exists (to avoid double-tagging)
            already_tagged = any(marker in content[:50] for marker in ["📄", "🌐", "💭", "🐍"])

            if not already_tagged:
                # Force add the appropriate tag
                if source_used == "documents":
                    response.content = "📄 **[Source: Your Documents]**\n\n" + content
                elif source_used == "web":
                    response.content = "🌐 **[Source: Web Search]**\n\n" + content
                elif source_used == "both":
                    response.content = "📄🌐 **[Source: Documents + Web]**\n\n" + content
                else:
                    response.content = "💭 **[Source: AI Knowledge]**\n\n" + content

    return {'messages': [response]}


def web_retrieve_node(state: GraphState):
    """
    Retrieves web results and adds them to state.
    """
    print("---WEB SEARCH---")
    question = state["messages"][-1].content

    try:
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
    except Exception as e:
        print(f"⚠️ Web search error: {e}")
        return {"web_documents": []}


def should_continue(state: GraphState):
    messages = state['messages']
    last_message = messages[-1]

    # If there are tool calls, route to the tool node
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"

    # Check if this response came after a tool execution
    # Look back through messages to see if any tools were called
    if len(messages) >= 2:
        # Check if we just came from tools
        for i in range(len(messages) - 2, max(0, len(messages) - 5), -1):
            msg = messages[i]
            # Check if this message has tool calls
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                # This response is after a tool call, tag it
                if hasattr(last_message, 'content') and last_message.content:
                    content = last_message.content
                    # Check if not already tagged
                    if "🐍" not in content[:50]:
                        last_message.content = "🐍 **[Source: Python Code Execution]**\n\n" + content
                break

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
from fastapi import FastAPI
app = FastAPI()

@app.post("/chat")
def chat(request: dict):
    print("Received:", request)
    user_input = request.get("message")
    session_id = request.get("session_id") or request.get("thread_id") or "default-thread"

    config = {"configurable": {"thread_id": session_id}}

    try:
        # Pass session_id in the state
        result = graph.invoke({
            "messages": [HumanMessage(content=user_input)],
            "session_id": session_id
        }, config)
        response = result["messages"][-1].content
        return {"response": response}
    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        return {"response": f"Sorry, an error occurred: {str(e)}"}

