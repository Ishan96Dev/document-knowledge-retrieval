"""
CrewAI-style Agents for document retrieval and response generation.
Uses direct OpenAI API calls since CrewAI doesn't support Python 3.14 yet.

============================================================
Created by: Ishan Chakraborty
License: MIT License
Copyright (c) 2024 Ishan Chakraborty
============================================================
"""
from openai import OpenAI
from typing import List, Dict, Any, Optional

import config


class Agent:
    """A simple agent class that wraps OpenAI API calls."""
    
    def __init__(self, role: str, goal: str, backstory: str, model: Optional[str] = None):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = model or config.OPENAI_MODEL_NAME
    
    def set_model(self, model: str):
        """Update the model used by this agent."""
        self.model = model
    
    def run(self, task: str, context: str = "") -> str:
        """Execute a task using the agent's persona."""
        system_prompt = f"""You are a {self.role}.

Your goal: {self.goal}

Background: {self.backstory}

Always provide helpful, accurate, and well-structured responses."""

        user_message = task
        if context:
            user_message = f"{task}\n\nContext:\n{context}"
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        return response.choices[0].message.content


class RetrievalAgent(Agent):
    """Agent specialized in analyzing and prioritizing retrieved information."""
    
    def __init__(self):
        super().__init__(
            role="Document Retrieval Specialist",
            goal="Find and retrieve the most relevant document chunks to answer user queries accurately",
            backstory="""You are an expert at understanding user queries and finding the most 
            relevant information from a knowledge base. You excel at semantic search and 
            understanding context. You always strive to find the best matching documents 
            that will help answer the user's question comprehensively."""
        )
    
    def analyze(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Analyze retrieved chunks and identify the most relevant information."""
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            context_parts.append(
                f"[Source {i+1}: {chunk['source']}, Page {chunk.get('page', 'N/A')}]\n{chunk['text']}"
            )
        
        context = "\n\n---\n\n".join(context_parts)
        
        task = f"""Analyze the following retrieved document chunks to answer this query:

Query: {query}

Your job is to:
1. Identify which chunks are most relevant to the query
2. Extract the key information that answers the question
3. Note any gaps or missing information
4. Organize the relevant information logically"""

        return self.run(task, context)


class ResponseAgent(Agent):
    """Agent specialized in synthesizing final responses."""
    
    def __init__(self):
        super().__init__(
            role="Knowledge Synthesizer",
            goal="Synthesize retrieved information into clear, accurate, and helpful responses",
            backstory="""You are a skilled communicator who excels at understanding complex 
            information and presenting it in a clear, accessible way. You always cite your 
            sources and provide accurate information based on the retrieved documents. 
            You are careful to only provide information that is supported by the sources 
            and clearly indicate when information might be incomplete or uncertain."""
        )
    
    def synthesize(self, query: str, analysis: str, sources: List[Dict[str, Any]]) -> str:
        """Synthesize the final response based on analysis."""
        source_list = ", ".join([s['source'] for s in sources[:5]])
        
        task = f"""Based on the analysis provided, create a comprehensive response to the user's query.

Query: {query}

Analysis of retrieved documents:
{analysis}

Available sources: {source_list}

Your response should:
1. Directly answer the user's question
2. Include relevant details and context
3. Cite sources when mentioning specific information (e.g., "According to document.pdf...")
4. Be clear and well-organized
5. Acknowledge if the answer is incomplete or uncertain based on available information"""

        return self.run(task)


class AnalyzerAgent(Agent):
    """Agent specialized in document analysis and summarization."""
    
    def __init__(self):
        super().__init__(
            role="Document Analyzer",
            goal="Analyze documents to extract key insights, summaries, and important points",
            backstory="""You are an expert at analyzing documents and extracting valuable 
            insights. You can identify key themes, important facts, and summarize complex 
            content effectively. You help users understand the overall content and structure 
            of their documents."""
        )
    
    def analyze_document(self, document_texts: List[str], source_name: str) -> str:
        """Analyze a document and provide a summary."""
        combined_text = "\n\n---\n\n".join(document_texts[:5])
        
        task = f"""Analyze this document and provide a comprehensive summary:

Document: {source_name}

Content:
{combined_text}

Provide:
1. Main topics and themes
2. Key facts and takeaways
3. Brief overall summary
4. Notable insights or important details"""

        return self.run(task)
