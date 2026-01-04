## PolicyBot

PolicyBot is a Retrieval-Augmented Generation (RAG) based assistant designed to provide precise, context-aware answers to organizational internal document queries. It utilizes advanced retrieval techniques and a specialized inference engine to ensure responses are strictly grounded in authorized internal policies and Standard Operating Procedures (SOPs).

### Features

*   **RAG-Driven Architecture**: Combines vector-based retrieval with generative AI to answer queries based exclusively on provided context.
*   **Contextual Compression & Reranking**: Utilizes FlashrankRerank to refine large sets of retrieved documents down to the most relevant content before generation.
*   **Ambiguity Resolution**: Features a built-in AMBIGUITY\_MAP to handle overlapping procedures for different roles, such as Faculty vs. Research Scholars.
*   **Dual Interface Support**: Includes a FastAPI backend (app.py) for web integration and a terminal-based chat interface (chat.py) for debugging.
*   **Strict Formatting**: Automatically formats responses with bold key terms, numbered lists for procedures, and clear distinctions for specialized reports.

### Project Structure

PolicyBot/

├── chroma\_db/ # Persistent vector store directory

├── utils/

│ ├── engine.py # Core RAG chain logic and retrieval engine

│ ├── chat.py # Terminal-based chat interface with debug logs

├── app.py # FastAPI backend entry point

├── ingest.py # Document processing and vector indexing

├── .env # Environment variables (API Keys)

├── Dockerfile # Containerization configuration

└── requirements.txt # Project dependencies

### 

### Installation

#### 1\. Clone the Repository

git clone https://github.com/yourusername/PolicyBot.git

cd PolicyBot

#### 2\. Set Up Environment Variables

Create a .env file in the root directory:

OPENROUTER\_API\_KEY=your\_api\_key\_here

#### 3\. Install Dependencies

pip install -r requirements.txt

python -m spacy download en\_core\_web\_sm

### Usage

#### 1\. Ingest Documents

Run the ingestion script to process your documents into the ChromaDB vector store:

python ingest.py

#### 

#### 2\. Run the API (Production/Web)

Start the FastAPI backend:

uvicorn app:app --reload

#### 

#### 3\. Terminal Chat (Debugging)

To test the engine in the terminal with source document tracking and chunk logs:

python chat.py

### Technical Methodology

The engine.py logic implements a sophisticated RAG pipeline:

1.  **Retrieval**: The system performs an initial search against a ChromaDB collection using HuggingFaceEmbeddings, retrieving the top 25 potential matches.
2.  **Compression**: The FlashrankRerank model (ms-marco-MiniLM-L-12-v2) evaluates those 25 candidates and selects the top 5 most relevant chunks to reduce noise.
3.  **Inference**: The query and compressed context are sent to a ChatOpenAI model via OpenRouter. The model is configured with a temperature of 0.0 to ensure deterministic, factual output.
4.  **Strict Grounding**: The system is prompted to answer strictly using the provided context. If no relevant answer is found, it provides a standard fallback message.

### Contributing

Internal contributions should follow the standard fork-and-pull-request workflow. Ensure that any changes to the prompt templates in engine.py maintain the required formatting rules for industrial reports and procedures.
