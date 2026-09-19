# retriever.py
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any, Optional  # ◄── UPDATE: Added Optional here

# Initialize the Cross-Encoder ranking judge once
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

class ReRankingRetriever(BaseRetriever):
    vectorstore: Any
    k: int = 3
    # ◄── FIX: Wrap with Optional so Pydantic accepts NoneType values cleanly
    search_filter: Optional[Dict[str, Any]] = None 

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        
        # 1. Retrieve more candidates than needed (k * 3) using metadata filters if present
        initial_results = self.vectorstore.similarity_search(
            query, 
            k=self.k * 3,
            filter=self.search_filter
        )
        
        if not initial_results:
            return []

        # 2. Pair query with contents for Cross-Encoder scoring
        pairs = [[query, doc.page_content] for doc in initial_results]
        scores = reranker.predict(pairs)
        
        # 3. Zip and sort documents by relevance score (highest first)
        ranked_results = sorted(
            zip(initial_results, scores),
            key=lambda x: x[1],  # Sorted by score index
            reverse=True
        )
        
        # 4. Extract and slice out top K items
        return [doc for doc, score in ranked_results[:self.k]]
