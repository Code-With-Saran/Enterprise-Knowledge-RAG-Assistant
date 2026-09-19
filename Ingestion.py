# ingestion.py
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

DATA_DIR = "./data"
DB_DIR = "./chroma_db"

def run_ingestion():
    print("🔄 Step 1: Starting document ingestion...")
    
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        raise FileNotFoundError(f"❌ The directory '{DATA_DIR}' is empty or missing. Please add your .txt files.")

    # 1. Load all .txt files from the directory
    loader = DirectoryLoader(path=DATA_DIR, glob="*.txt", loader_cls=TextLoader)
    raw_documents = loader.load()
    print(f"📄 Loaded {len(raw_documents)} raw document source files.")

    # 2. Add custom catalog metadata based on the file name dynamically
    for doc in raw_documents:
        source_name = doc.metadata.get("source", "").lower()
        if "financial" in source_name:
            doc.metadata["department"] = "Finance"
        elif "policy" in source_name:
            doc.metadata["department"] = "HR"
        else:
            doc.metadata["department"] = "General"

    # 3. Chunk documents intelligently
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(raw_documents)
    print(f"✂️ Sliced documents into {len(chunks)} text chunks.")

    print("🔄 Step 2: Generating vector embeddings and saving to ChromaDB...")
    # 4. Initialize embedding model
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # 5. Build and persist the vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    print("💾 Vector store saved successfully to disk.")
    return vectorstore

def load_existing_vectorstore():
    """Loads the database from the disk if ingestion was already run."""
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists(DB_DIR):
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    return None
