"""
Crew orchestration for document retrieval workflow.
Uses custom agent implementation for Python 3.14 compatibility.

============================================================
Created by: Ishan Chakraborty
License: MIT License
Copyright (c) 2024 Ishan Chakraborty
============================================================
"""
from typing import List, Dict, Any, Optional

from src.agents import RetrievalAgent, ResponseAgent, AnalyzerAgent
from src.vector_store import VectorStoreManager


class DocumentRAGCrew:
    """Orchestrates the multi-agent RAG workflow."""
    
    def __init__(self, vector_store: VectorStoreManager, model: Optional[str] = None):
        self.vector_store = vector_store
        self.retrieval_agent = RetrievalAgent()
        self.response_agent = ResponseAgent()
        self.analyzer_agent = AnalyzerAgent()
        if model:
            self.set_model(model)
    
    def set_model(self, model: str):
        """Update model for all agents."""
        self.retrieval_agent.set_model(model)
        self.response_agent.set_model(model)
        self.analyzer_agent.set_model(model)
    
    def query(self, user_query: str, top_k: int = 5) -> Dict[str, Any]:
        """Process a user query through the RAG pipeline."""
        # Step 1: Retrieve relevant documents
        search_results = self.vector_store.search(user_query, top_k=top_k)
        
        if not search_results:
            return {
                "answer": "I couldn't find any relevant information in the uploaded documents. Please make sure you've uploaded documents related to your query.",
                "sources": [],
                "success": False
            }
        
        # Step 2: Format sources
        sources = []
        for result in search_results:
            sources.append({
                "source": result["source"],
                "page": result.get("page", 0),
                "score": result["score"],
                "text": result["text"][:300] + "..." if len(result["text"]) > 300 else result["text"]
            })
        
        # Step 3: Retrieval agent analyzes the chunks
        analysis = self.retrieval_agent.analyze(user_query, search_results)
        
        # Step 4: Response agent synthesizes the final answer
        response = self.response_agent.synthesize(user_query, analysis, sources)
        
        return {
            "answer": response,
            "sources": sources,
            "success": True
        }
    
    def analyze_document(self, document_chunks: List[Any], source_name: str) -> str:
        """Analyze a newly uploaded document."""
        texts = [chunk.page_content for chunk in document_chunks[:5]]
        return self.analyzer_agent.analyze_document(texts, source_name)
    
    def simple_query(self, user_query: str, top_k: int = 5) -> Dict[str, Any]:
        """Process a query with just retrieval (no agents) for faster response."""
        search_results = self.vector_store.search(user_query, top_k=top_k)
        
        if not search_results:
            return {
                "answer": "No relevant documents found.",
                "sources": [],
                "success": False
            }
        
        # Format sources
        sources = []
        context_parts = []
        
        for result in search_results:
            sources.append({
                "source": result["source"],
                "page": result.get("page", 0),
                "score": result["score"],
                "text": result["text"][:500] + "..." if len(result["text"]) > 500 else result["text"]
            })
            context_parts.append(result["text"])
        
        return {
            "context": "\n\n".join(context_parts),
            "sources": sources,
            "success": True
        }
