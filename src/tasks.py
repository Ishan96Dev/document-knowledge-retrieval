"""
CrewAI Tasks for document retrieval workflow.
"""
from crewai import Task
from typing import List, Dict, Any


def create_retrieval_task(agent, query: str, context: str) -> Task:
    """Create a task for retrieving relevant documents."""
    return Task(
        description=f"""Analyze the user's query and the retrieved document chunks to determine 
        which information is most relevant.
        
        User Query: {query}
        
        Retrieved Context:
        {context}
        
        Your job is to:
        1. Understand what the user is asking
        2. Identify the most relevant pieces of information from the context
        3. Organize the relevant information logically
        4. Note any gaps in the available information
        """,
        expected_output="""A structured analysis containing:
        - The key points relevant to the query
        - Organized and prioritized information
        - Any limitations or gaps in the available information""",
        agent=agent
    )


def create_response_task(agent, query: str, retrieval_output: str) -> Task:
    """Create a task for synthesizing the final response."""
    return Task(
        description=f"""Using the analyzed information, create a comprehensive and helpful 
        response to the user's query.
        
        User Query: {query}
        
        Analyzed Information:
        {retrieval_output}
        
        Your job is to:
        1. Synthesize the information into a clear, coherent response
        2. Directly answer the user's question
        3. Include relevant details and context
        4. Cite sources when mentioning specific information
        5. Indicate if the answer is incomplete or uncertain
        """,
        expected_output="""A well-structured response that:
        - Directly answers the user's query
        - Provides relevant supporting details
        - Is clear and easy to understand
        - Cites sources appropriately
        - Acknowledges any limitations""",
        agent=agent
    )


def create_analysis_task(agent, document_texts: List[str], source_name: str) -> Task:
    """Create a task for analyzing uploaded documents."""
    combined_text = "\n\n---\n\n".join(document_texts[:5])  # Limit to first 5 chunks
    
    return Task(
        description=f"""Analyze the following document content and provide a comprehensive summary.
        
        Document: {source_name}
        
        Content:
        {combined_text}
        
        Your job is to:
        1. Identify the main topics and themes
        2. Extract key facts and information
        3. Summarize the document's purpose and content
        4. Note any important details or insights
        """,
        expected_output="""A document analysis containing:
        - Main topics and themes
        - Key facts and takeaways
        - Brief summary of the document
        - Notable insights or important details""",
        agent=agent
    )
