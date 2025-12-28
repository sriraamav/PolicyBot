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
    "CAT QP": {
        "prompt": "I found different procedures for Course Coordinator and Course Handling Faculty. Which are you?",
        "options": {"1": "Course Coordinator", "2": "Course Handling Faculty"},
        "tags": ["coordinator", "faculty"]
    }
}

logging.getLogger("httpx").setLevel(logging.WARNING)

def chat():
    print("🔹 Loading vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

 
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=4)

    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )

    
    llm = ChatOpenAI(
        model="nvidia/nemotron-3-nano-30b-a3b:free", 
        openai_api_key=OS_OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "http://localhost:3000", 
            "X-Title": "Local SOP Bot",             
        },
        temperature=0.1,
        streaming=True 
    )

    prompt = PromptTemplate.from_template(
        """
    You are a retrieval-based assistant.
    You MUST answer strictly using the provided context.
    At the end of your answer, you can add any relevant information from the documents as additional context.
    If the answer is not present in the context, say:
    "I couldn't find a relevant answer. Please be more specific."

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

    print("\nRAG Chatbot ready (type 'exit' to quit)\n")

    while True:
        query = input("Query: ").strip()
        if not query: continue
        if query.lower() == "exit": break

        user_input = query      
        for topic, data in AMBIGUITY_MAP.items():
            if topic in user_input.lower():
                if not any(tag in user_input.lower() for tag in data["tags"]):
                    print(f"\n{data['prompt']}")
                    for k, v in data["options"].items():
                        print(f"   {k}. {v}")
                    
                    choice = input("\nEnter number: ").strip()
                    selected = data["options"].get(choice)
                    if selected:
                        user_input += f" for {selected}"
                        print(f"Narrowing search to: {selected}\n")
                    break

        print("\nAnswer: ", end="", flush=True)

        full_answer = ""
        for chunk in rag_chain.stream(user_input):
            print(chunk, end="", flush=True)
            full_answer += chunk

        print("\n" + "-" * 60)
        
        docs = retriever.invoke(user_input)
        print(f"\n[DEBUG] Found {len(docs)} relevant SOP sections.")
        for i, d in enumerate(docs):
            source = d.metadata.get('source', 'Unknown')
            print(f"   - Doc {i+1}: {os.path.basename(source)}")




if __name__ == "__main__":
    chat()

    