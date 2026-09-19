# Pipeline.py
import os  
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain
from Retriever import ReRankingRetriever

def build_rag_pipeline(vectorstore, target_department=None, top_k=3):
    print("⚙️ Step 3: Assembling RAG Pipeline using Native Groq Cloud...")
    
    # Fetch the key string explicitly from the active environment cache
    api_key_string = os.getenv("OPENAI_API_KEY")
    
    # Safety Check
    if not api_key_string:
        raise ValueError("❌ Critical Error: OPENAI_API_KEY was not loaded into memory! Check your .env setup.")

    # 1. ◄── CHANGED: Updated model_name to a fully active, universally supported text model ID
    llm = ChatGroq(
        temperature=0, 
        model_name="openai/gpt-oss-120b",  # Active open-source production text model on Groq
        groq_api_key=api_key_string  
    )
    
    # Build metadata filter configuration if specified
    search_filter = {"department": target_department} if target_department else None

    # Initialize your custom re-ranking retriever
    custom_retriever = ReRankingRetriever(
        vectorstore=vectorstore,
        k=top_k,
        search_filter=search_filter
    )

    # Build standard Conversational Chain with Memory support
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=custom_retriever,
        return_source_documents=True
    )
    print("🚀 Conversational Groq Pipeline compiled successfully.")
    return qa_chain
