# Enterprise-Knowledge-RAG-Assistant
A modular RAG pipeline built with LangChain and Streamlit for multi-file .txt analysis. Powered by Groq ChatGroq for high-speed reasoning, it includes a Cross-Encoder re-ranker for search precision. Features dynamic metadata filtering, conversational memory for continuous chat, and an interactive sidebar to hot-swap files.
# Enterprise Knowledge RAG Assistant 🤖💼

A highly optimized, production-ready **Retrieval-Augmented Generation (RAG)** pipeline designed for intelligent multi-file text analysis. This project features a modular architecture built entirely in Python using **LangChain-Classic**, **Streamlit**, and **ChromaDB**, and it is powered by **Groq Cloud's** ultra-fast inference infrastructure.

---

## 🚀 Key Features

*   **Modular Architecture:** Perfectly separated files (`ingestion.py`, `retriever.py`, `pipeline.py`, `main.py`) ensuring clean updates and scalable future modifications.
*   **Advanced Re-ranking (Cross-Encoder):** Casts a wide net using traditional vector search, then utilizes a heavy-duty `CrossEncoder` model to score and re-sort documents for maximum answer accuracy.
*   **Metadata Tagging & Pre-Filtering:** Automatically categorizes text documents during ingestion and applies explicit database filters to block old or irrelevant records before computing vector scores.
*   **Conversational Memory Tracker:** Remembers chat context across queries, allowing the AI to naturally process complex follow-up questions.
*   **Interactive Streamlit Dashboard:** A web interface featuring real-time ChatGPT-style chat elements, expanding reference document citations, and an asset sidebar.
*   **Hot-Swapping File Controls:** Features a dedicated browser drag-and-drop file uploader and a **"Clear Database & Start Fresh"** master button to instantly wipe database partitions and load new document batches.

---

## 📂 Project Architecture Structure

```text
rag_project/
│
├── data/                       # Drop your raw source .txt files here
├── chroma_db/                  # Local vector database directory (Auto-created)
│
├── ingestion.py                # Pipeline Step 1 & 2: Handles file loading, structural chunking, and storage
├── retriever.py                # Custom Component: Cross-Encoder judge setup with metadata rules
├── pipeline.py                 # Pipeline Step 3: Compiles the conversational QA engine
├── main.py                     # Entry Point: Orchestrates Streamlit frontend views and application loops
│
├── .env                        # Secure configuration containing your API pass keys
└── requirements.txt            # System dependencies manifest
```

---

## 🛠️ Step-by-Step Installation

### 1. Clone the Repository
```bash
git clone https://github.com
cd your-repo-name
```

### 2. Install Required Dependencies
Ensure your development environment is using Python 3.10+ and install all required packages at once:
```bash
pip install -r requirements.txt
```
*(If you haven't created a requirements file yet, use: `pip install streamlit langchain langchain-community langchain-groq sentence-transformers chromadb python-dotenv pypdf langchain-text-splitters langchain-classic`)*

### 3. Setup Environment Variables
Create a file named `.env` in the root project folder and securely paste your free Groq API key:
```text
OPENAI_API_KEY=gsk_your_actual_groq_cloud_api_key_here
```

---

## 🏃 Running the Application

1. Open your terminal inside the project directory in VS Code.
2. Drop some seed text files inside the `./data` folder to let the pipeline initialize.
3. Launch your web engine dashboard using the Streamlit module wrapper:
   ```bash
   python -m streamlit run main.py
   ```
4. A browser window will automatically open up at `http://localhost:8501`. You can now chat continuously, view source text clippings, and upload new documents on demand!
