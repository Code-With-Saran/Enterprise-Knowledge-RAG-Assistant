# main.py
import sys
import os
from dotenv import load_dotenv
# 1. MUST LOAD ENVIRONMENT VARIABLES FIRST
load_dotenv() 

# 2. Add system path and continue loading other modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from Ingestion import run_ingestion, load_existing_vectorstore
from Pipeline import build_rag_pipeline

# 1. Page Configuration Setup
st.set_page_config(page_title="Enterprise RAG Assistant", page_icon="🤖", layout="wide")
st.title("🤖 Enterprise Knowledge RAG Assistant")
st.caption("A modular RAG system featuring cross-encoder re-ranking, metadata filtering, and chat memory.")

# 2. Securely load environmental API Keys
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ OPENAI_API_KEY is missing! Please configure your .env file in VS Code.")
    st.stop()

# 3. Cache the Core Database and Pipeline Engine (Loads ONLY once)
@st.cache_resource
def initialize_engine():
    vectorstore = load_existing_vectorstore()
    if not vectorstore:
        vectorstore = run_ingestion()
    return vectorstore

# Initialize or load our persistent database connection
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = initialize_engine()

# Always compile the QA pipeline using the current database instance from session state
qa_chain = build_rag_pipeline(st.session_state.vectorstore, target_department=None, top_k=3)

# =====================================================================
# 📁 SIDEBAR FILE UPLOADER COMPONENT
# =====================================================================
with st.sidebar:
    st.header("📂 Document Management")
    st.write("Upload new `.txt` files directly into your active RAG database.")
    
    # 1. Create file uploader box (Accepts multiple files)
    uploaded_files = st.file_uploader(
        "Choose Text Files", 
        type=["txt"], 
        accept_multiple_files=True
    )
    
    # 2. Process uploads when button is clicked
    if st.button("🚀 Process and Inject Documents"):
        if uploaded_files:
            new_chunks_count = 0
            
            with st.spinner("Processing documents into vector store..."):
                for uploaded_file in uploaded_files:
                    # Read the raw string data out of the uploaded browser cache
                    string_data = uploaded_file.read().decode("utf-8")
                    
                    # Convert the raw text string into a standardized LangChain Document object
                    # We inject custom metadata tracking file names and tags
                    doc = Document(
                        page_content=string_data,
                        metadata={
                            "source": uploaded_file.name,
                            "department": "General" # Default classification tag
                        }
                    )
                    
                    # Cut the document into standard uniform pieces
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                    file_chunks = text_splitter.split_documents([doc])
                    
                    # Push these new text chunks instantly into our operational Chroma database
                    st.session_state.vectorstore.add_documents(file_chunks)
                    new_chunks_count += len(file_chunks)
            
            st.success(f"✅ Injected {len(uploaded_files)} files ({new_chunks_count} text chunks) successfully!")
            st.rerun() # Refresh the web browser page state to update search parameters
        else:
            st.warning("Please upload at least one `.txt` file first.")

# =====================================================================
# 💬 THE CHAT USER INTERFACE
# =====================================================================

# Initialize Streamlit Persistent Memory
if "messages" not in st.session_state:
    st.session_state.messages = []  # Holds UI chat bubbles
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Holds raw data tuple memory for LangChain

# Display existing message logs on browser refresh
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📚 View Supporting Sources Used"):
                st.markdown(message["sources"])

# Handle New Chat Inputs from the user
if user_query := st.chat_input("Ask a question about your documents..."):
    
    # Display the user's typed question instantly on screen
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Trigger Assistant Response window
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking and researching sources... 🔍")
        
        try:
            # Pass the question AND history into the memory chain
            response = qa_chain({
                "question": user_query, 
                "chat_history": st.session_state.chat_history
            })
            
            answer = response["answer"]
            source_docs = response.get("source_documents", [])

            # Format source document metadata visually
            sources_md = ""
            for i, doc in enumerate(source_docs, 1):
                file_origin = doc.metadata.get('source', 'Unknown File')
                dept_tag = doc.metadata.get('department', 'None')
                snippet = doc.page_content[:150].replace('\n', ' ')
                sources_md += f"**[{i}] File:** `{file_origin}` (Dept: *{dept_tag}*)\n> {snippet}...\n\n"

            # Display the generated content cleanly
            message_placeholder.markdown(answer)
            if sources_md:
                with st.expander("📚 View Supporting Sources Used"):
                    st.markdown(sources_md)

            # Save the execution details into persistent session states
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer, 
                "sources": sources_md
            })
            
            # Update LangChain memory structure with the latest exchange tuple
            st.session_state.chat_history.append((user_query, answer))

        except Exception as e:
            st.error(f"An execution error occurred: {e}")
