import os
import tempfile
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()


class UserDocumentHandler:
    def __init__(self):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment")
        self.embeddings = OpenAIEmbeddings()
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        self.collection_name = "my_docs"

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?"]
        )

    def process_uploaded_file(self, file_bytes: bytes, filename: str, session_id: str) -> List[Document]:
        """Process uploaded file and return document chunks with session metadata."""

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name

        try:
            # Load based on file type
            if filename.lower().endswith('.pdf'):
                loader = PyPDFLoader(tmp_path)
            elif filename.lower().endswith('.txt'):
                loader = TextLoader(tmp_path)
            elif filename.lower().endswith('.docx'):
                loader = Docx2txtLoader(tmp_path)
            else:
                raise ValueError(f"Unsupported file type: {filename}")

            # Load and split
            docs = loader.load()
            chunks = self.text_splitter.split_documents(docs)

            # Add session metadata to each chunk
            for chunk in chunks:
                chunk.metadata["session_id"] = session_id
                chunk.metadata["source_file"] = filename
                chunk.metadata["user_uploaded"] = True

            return chunks

        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def upload_to_qdrant(self, chunks: List[Document]) -> int:
        """Upload document chunks to Qdrant and return count."""
        Qdrant.from_documents(
            chunks,
            self.embeddings,
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=self.collection_name,
            force_recreate=False,
        )
        return len(chunks)
