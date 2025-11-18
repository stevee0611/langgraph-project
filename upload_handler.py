import os
import tempfile
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

load_dotenv()

class UserDocumentHandler:
    def __init__(self):
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=60,
            prefer_grpc=False,
        )
        self.collection_name = "my_docs"
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?"]
        )
        self._embeddings = None
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        try:
            self.qdrant_client.get_collection(self.collection_name)
        except Exception:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
            )

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings()
        return self._embeddings

    def process_uploaded_file(self, file_bytes: bytes, filename: str, session_id: str) -> List[Document]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name

        try:
            if filename.lower().endswith('.pdf'):
                loader = PyPDFLoader(tmp_path)
            elif filename.lower().endswith('.txt'):
                loader = TextLoader(tmp_path)
            elif filename.lower().endswith('.docx'):
                loader = Docx2txtLoader(tmp_path)
            else:
                raise ValueError(f"Unsupported file type: {filename}")

            docs = loader.load()
            chunks = self.text_splitter.split_documents(docs)

            for chunk in chunks:
                chunk.metadata["session_id"] = session_id
                chunk.metadata["source_file"] = filename
                chunk.metadata["user_uploaded"] = True

            return chunks
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def upload_to_qdrant(self, chunks: List[Document]) -> int:
        QdrantVectorStore.from_documents(
            chunks,
            self.embeddings,
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=self.collection_name,
        )
        return len(chunks)

