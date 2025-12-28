import os
import shutil
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

DATA_DIR = "./data"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "documents"


def load_documents():
    docs = []

    for file in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, file)

        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())

        elif file.endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
            docs.extend(loader.load())

    return docs


def ingest():
    if os.path.exists(CHROMA_DIR):
        print("Clearing old database...")
        shutil.rmtree(CHROMA_DIR)

    print("🔹 Loading documents...")
    documents = load_documents()

    if not documents:
        raise ValueError("No documents found in ./data")

    print(f"✓ Loaded {len(documents)} documents")

    print("🔹 Chunking documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300,
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print("🔹 Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    print("🔹 Storing in ChromaDB...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )

    print("Ingestion complete!")


if __name__ == "__main__":
    ingest()
