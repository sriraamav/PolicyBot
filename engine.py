import os
import logging
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank

logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()
OS_OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "documents"

AMBIGUITY_MAP = {
    "conference": {
        "prompt": "I found different procedures for Faculty and Research Scholars. Which are you?",
        "options": {"1": "Faculty", "2": "Research Scholar"},
        "tags": ["faculty", "scholar", "researcher"]
    },
    "project": {
        "prompt": "Are you asking about Review Panel instructions or Guide responsibilities?",
        "options": {"1": "Review Panel Member", "2": "Project Guide"},
        "tags": ["panel", "member", "guide"]
    },
    "question paper": {
        "prompt": "I found different procedures for Course Coordinator and Course Handling Faculty. Which are you?",
        "options": {"1": "Course Coordinator", "2": "Course Handling Faculty"},
        "tags": ["coordinator", "faculty"]
    }
}

def get_rag_chain():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

 
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 25})

    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=5)

    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )

    
    llm = ChatOpenAI(
        model="xiaomi/mimo-v2-flash:free", 
        openai_api_key=OS_OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "http://localhost:3000", 
            "X-Title": "Local SOP Bot",             
        },
        temperature=0.0,
        streaming=True 
    )

    prompt = PromptTemplate.from_template(
        """
    You are a retrieval-based assistant.
    You MUST answer strictly using the provided context.
    At the end of your answer, you can add any relevant information from the documents as additional context.
    If the answer is not present in the context, say:
    "I couldn't find a relevant answer. Please be more specific."

    Strictly follow these formatting rules:
    1. Use **bold** for key terms and steps.
    2. Use bullet points or numbered lists for procedures.
    3. Use a new line for every new step.
    4. Ensure industrial visit reports are clearly distinct from other events.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain