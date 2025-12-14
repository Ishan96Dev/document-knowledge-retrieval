"""
Vector store module using Milvus (Zilliz Cloud) for document embeddings storage and retrieval.

============================================================
Created by: Ishan Chakraborty
License: MIT License
Copyright (c) 2024 Ishan Chakraborty
============================================================
"""
from typing import List, Dict, Any, Optional
from openai import OpenAI
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema
from langchain_core.documents import Document

import config


class VectorStoreManager:
    """Manages Milvus vector store operations with OpenAI embeddings."""
    
    def __init__(self):
        self.client = None
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.collection_name = config.MILVUS_COLLECTION_NAME
        self.embedding_model = config.OPENAI_EMBEDDING_MODEL
        self.embedding_dimension = config.EMBEDDING_DIMENSION
        self._connect()
    
    def _connect(self):
        """Connect to Milvus/Zilliz Cloud."""
        self.client = MilvusClient(
            uri=config.MILVUS_URI,
            token=config.MILVUS_TOKEN
        )
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Ensure the collection exists with proper schema."""
        if not self.client.has_collection(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=self.embedding_dimension,
                metric_type="COSINE",
                auto_id=True
            )
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a text using OpenAI."""
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )
        return [item.embedding for item in response.data]
    
    def add_documents(self, documents: List[Document]) -> int:
        """Add documents to the vector store."""
        if not documents:
            return 0
        
        # Extract texts and metadata
        texts = [doc.page_content for doc in documents]
        
        # Generate embeddings in batches
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.get_embeddings_batch(batch)
            all_embeddings.extend(embeddings)
        
        # Prepare data for Milvus
        data = []
        for i, doc in enumerate(documents):
            data.append({
                "vector": all_embeddings[i],
                "text": doc.page_content[:65535],  # Milvus varchar limit
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", 0),
                "chunk_index": doc.metadata.get("chunk_index", 0)
            })
        
        # Insert into Milvus
        self.client.insert(
            collection_name=self.collection_name,
            data=data
        )
        
        return len(data)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        query_embedding = self.get_embedding(query)
        
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=top_k,
            output_fields=["text", "source", "page", "chunk_index"]
        )
        
        # Format results
        formatted_results = []
        if results and len(results) > 0:
            for hit in results[0]:
                formatted_results.append({
                    "text": hit["entity"].get("text", ""),
                    "source": hit["entity"].get("source", "unknown"),
                    "page": hit["entity"].get("page", 0),
                    "chunk_index": hit["entity"].get("chunk_index", 0),
                    "score": hit["distance"]
                })
        
        return formatted_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            # Try to get count by querying
            if self.client.has_collection(self.collection_name):
                # Use num_entities to get count
                stats = self.client.get_collection_stats(self.collection_name)
                # Handle different response formats
                if isinstance(stats, dict):
                    row_count = stats.get("row_count", 0)
                    if row_count == 0:
                        row_count = stats.get("data", {}).get("row_count", 0)
                else:
                    row_count = 0
                return {
                    "row_count": row_count,
                    "collection_name": self.collection_name
                }
            return {"row_count": 0, "collection_name": self.collection_name}
        except Exception as e:
            return {"row_count": 0, "error": str(e)}
    
    def clear_collection(self):
        """Clear all documents from the collection."""
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            self._ensure_collection()
    
    def delete_by_source(self, source_name: str) -> int:
        """Delete all documents from a specific source."""
        try:
            result = self.client.delete(
                collection_name=self.collection_name,
                filter=f'source == "{source_name}"'
            )
            return result.get("delete_count", 0)
        except Exception as e:
            print(f"Error deleting documents: {e}")
            return 0
