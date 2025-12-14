"""
Document Knowledge Retrieval Tool - Streamlit Application
A multi-agent RAG system powered by Milvus and OpenAI.

============================================================
Created by: Ishan Chakraborty
License: MIT License
Copyright (c) 2024 Ishan Chakraborty
============================================================
"""
import streamlit as st
from pathlib import Path
import time
import base64
from openai import OpenAI

import config
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.crew import DocumentRAGCrew

# Available OpenAI Models
AVAILABLE_MODELS = {
    "gpt-4o": "GPT-4o (Latest)",
    "gpt-4o-mini": "GPT-4o Mini",
    "gpt-4-turbo": "GPT-4 Turbo",
    "o1-preview": "o1-preview",
    "o1-mini": "o1-mini"
}

# Page configuration
st.set_page_config(
    page_title="Document Knowledge Retrieval | Ishan Chakraborty",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean White Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Ensure material icons use the correct font */
    .material-icons {
        font-family: 'Material Icons', sans-serif !important;
    }
    
    .stApp {
        background: #f8f9fc !important;
    }
    
    .main .block-container {
        padding: 1.5rem 2rem !important;
        max-width: 1400px !important;
    }
    
    /* Header Styles */
    .main-header {
        color: #5046e5 !important;
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 0.25rem;
    }
    
    .sub-header {
        color: #666680 !important;
        text-align: center;
        font-size: 0.95rem !important;
        margin-bottom: 0.25rem;
    }
    
    .author-credit {
        color: #888 !important;
        text-align: center;
        font-size: 0.8rem !important;
        margin-bottom: 1rem;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e5e7eb !important;
        width: 400px !important;
        min-width: 400px !important;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #1a1a2e !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 10px !important;
        padding: 0.75rem !important;
    }
    
    [data-testid="stMetric"] label {
        color: #666680 !important;
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #5046e5 !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #ffffff !important;
        border-radius: 8px !important;
        padding: 0.25rem !important;
        border: 1px solid #e5e7eb !important;
        gap: 0.25rem !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #555566 !important;
        font-weight: 500 !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
        background: transparent !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #f3f4f6 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: #5046e5 !important;
        color: #ffffff !important;
    }
    
    .stTabs [aria-selected="true"] p {
        color: #ffffff !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: #5046e5 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }
    
    .stButton > button:hover {
        background: #4338ca !important;
        color: #ffffff !important;
    }
    
    .stButton > button p {
        color: #ffffff !important;
    }
    
    /* File list buttons small */
    .small-button > button {
        padding: 0.25rem 0.5rem !important;
        font-size: 0.75rem !important;
        min-height: 0px !important;
        height: auto !important;
    }
    
    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    /* Message Avatars */
    .stChatMessageAvatar {
        background-color: #e0e7ff !important;
        color: #5046e5 !important;
        font-size: 1.2rem !important;
        border: 1px solid #c7d2fe !important;
    }
    
    /* Source box */
    .source-box {
        background: #f8f9fc;
        border: 1px solid #e5e7eb;
        border-left: 3px solid #5046e5;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0 6px 6px 0;
        color: #333344;
    }
    
    /* Sidebar Alignment */
    div[data-testid="column"] button {
        margin: 0 auto;
        display: block;
    }
    
    /* Analytics Title Styling */
    .analytics-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 0.5rem;
    }
    
    /* Rephrase Box */
    .rephrase-box {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        color: #166534;
    }
    .rephrase-label {
        font-weight: 600;
        font-size: 0.85rem;
        color: #15803d;
        margin-bottom: 0.25rem;
    }
    .rephrase-text {
        font-size: 0.95rem;
    }
    
    hr {
        border-color: #e5e7eb !important;
        margin: 1rem 0 !important;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    # Logic to fetch initial DB stats safely
    db_chunks = 0
    
    defaults = {
        "messages": [],
        "document_processor": None,
        "vector_store": None,
        "crew": None,
        "initialized": False,
        "selected_model": "gpt-4o",
        "analytics": {
            "total_chunks": 0,
            "total_tokens_used": 0,
            "total_queries": 0,
            "estimated_cost": 0.0
        },
        "openai_client": None,
        "rephrase_query": "",
        "show_rephrased": False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    if st.session_state.document_processor is None:
        st.session_state.document_processor = DocumentProcessor()
    
    if st.session_state.openai_client is None and config.OPENAI_API_KEY:
        st.session_state.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)


def initialize_services():
    """Initialize vector store and crew."""
    if not st.session_state.initialized:
        is_valid, message = config.validate_config()
        if not is_valid:
            st.error(f"Configuration Error: {message}")
            return False
        
        try:
            with st.spinner("Connecting to Milvus..."):
                st.session_state.vector_store = VectorStoreManager()
                st.session_state.crew = DocumentRAGCrew(st.session_state.vector_store)
                st.session_state.initialized = True
                
                # Hydrate analytics from DB once connected
                stats = st.session_state.vector_store.get_collection_stats()
                st.session_state.analytics["total_chunks"] = stats.get("row_count", 0)
                
            return True
        except Exception as e:
            st.error(f"Failed to initialize: {str(e)}")
            return False
    return True


def update_analytics(chunks_added=0, tokens_used=0, query_made=False):
    """Update analytics counters."""
    st.session_state.analytics["total_chunks"] += chunks_added
    st.session_state.analytics["total_tokens_used"] += tokens_used
    if query_made:
        st.session_state.analytics["total_queries"] += 1
    
    embedding_cost = (st.session_state.analytics["total_chunks"] * 500 / 1000) * 0.00013
    llm_cost = (st.session_state.analytics["total_tokens_used"] / 1000) * 0.0004
    st.session_state.analytics["estimated_cost"] = embedding_cost + llm_cost


def rephrase_query_func(query: str) -> str:
    """Rephrase and expand query using OpenAI for better document retrieval."""
    if not query or not query.strip():
        st.warning("Please enter a query to rephrase.")
        return query
    
    try:
        client = st.session_state.openai_client
        if not client:
            st.error("OpenAI client not initialized.")
            return query
        
        with st.spinner("Enhancing your query..."):
            response = client.chat.completions.create(
                model=st.session_state.selected_model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert query optimizer for a RAG system.
Your task is to take a user's input and transform it into a highly effective, specific, and comprehensive search query for finding information in documents.

CRITICAL INSTRUCTION:
If the user provides a SHORT or VAGUE input (e.g., "Summarize", "Explain", "Key points"), you MUST expand it into a full, detailed instruction.

EXAMPLES:
Input: "Summarize"
Output: "Please provide a comprehensive summary of the document, detailing the main topics, key arguments, and primary conclusions."

Input: "Explain"
Output: "Explain the core concepts and ideas presented in this document in detail, providing context and examples if available."

Input: "Key points"
Output: "What are the most important key points, takeaways, and critical information mentioned in this document?"

Input: "Cost"
Output: "What specific information does the document provide regarding costs, pricing, expenses, or financial implications?"

Input: "How to fix deployment"
Output: "What are the step-by-step instructions or solutions provided in the document for troubleshooting and fixing deployment issues?"

Return ONLY the enhanced query. Do not add quotes or explanations."""
                    },
                    {"role": "user", "content": query}
                ],
                max_tokens=200,
                temperature=0.3
            )
            rephrased = response.choices[0].message.content.strip()
            return rephrased
    except Exception as e:
        st.error(f"Rephrase failed: {str(e)}")
        return query


def get_pdf_display(file_path: str) -> str:
    """Generate PDF display HTML using iframe."""
    try:
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()
        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
        return f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    except Exception as e:
        return f"<p>Error loading PDF: {str(e)}</p>"


def render_document_viewer(file_path: str, file_name: str):
    """Render document viewer."""
    path = Path(file_path)
    ext = path.suffix.lower()
    
    if ext == ".pdf":
        st.markdown(get_pdf_display(file_path), unsafe_allow_html=True)
    elif ext in [".txt", ".csv"]:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            st.text_area(f"Content of {file_name}", content, height=500, disabled=True)
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    else:
        st.warning(f"Preview not available for {ext} files.")


def render_sidebar():
    """Render sidebar with robust state and display logic."""
    with st.sidebar:
        st.markdown("### 📤 Document Upload")
        
        # Unique key for uploader to allow clearing if needed, though we rely on persistence check
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "txt", "csv"],
            accept_multiple_files=True,
            help="PDF, TXT, CSV"
        )
        
        # Check for new files that aren't already processed
        new_files = []
        if uploaded_files:
            existing_files = [f['name'] for f in st.session_state.document_processor.get_uploaded_files()]
            for f in uploaded_files:
                if f.name not in existing_files:
                    new_files.append(f)
        
        if new_files:
            if st.button("Process New Documents", use_container_width=True, type="primary"):
                process_uploaded_files(new_files)
        elif uploaded_files:
            st.info("All uploaded files are already processed.")
        
        st.markdown("---")
        
        # Model Selection
        st.markdown("### 🤖 Model Settings")
        selected = st.selectbox(
            "AI Model",
            options=list(AVAILABLE_MODELS.keys()),
            format_func=lambda x: AVAILABLE_MODELS[x],
            index=list(AVAILABLE_MODELS.keys()).index(st.session_state.selected_model)
        )
        if selected != st.session_state.selected_model:
            st.session_state.selected_model = selected
            if st.session_state.crew:
                st.session_state.crew.set_model(selected)
        
        st.markdown("---")
        
        # Uploaded Files
        st.markdown("### 📄 Uploaded Files")
        
        files = st.session_state.document_processor.get_uploaded_files()
        
        if files:
            for file in files:
                col1, col2, col3 = st.columns([0.60, 0.20, 0.20])
                with col1:
                    name = file['name']
                    if len(name) > 18:
                        name = name[:15] + "..."
                    st.markdown(f"<div style='padding-top: 5px;'><b>{name}</b></div>", unsafe_allow_html=True)
                    st.caption(f"{file['size']/1024:.1f} KB")
                with col2:
                    st.button("👁", key=f"view_{file['name']}", help="View Document", use_container_width=True, on_click=lambda f=file: st.session_state.update({"preview_file": f}))
                with col3:
                    if st.button("🗑", key=f"del_{file['name']}", help="Delete Document", use_container_width=True):
                        delete_file(file['name'])
        else:
            st.info("No documents uploaded.")
        
        st.markdown("---")
        
        # Knowledge Base
        st.markdown("### 🧠 Knowledge Base")
        # Use session state analytics for real-time consistency
        chunks_count = st.session_state.analytics["total_chunks"]
        st.metric("Indexed Chunks", chunks_count)
        
        st.markdown("---")
        
        # Actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear All", use_container_width=True):
                clear_all_data()
        with col2:
            if st.button("Reset DB", use_container_width=True):
                reset_milvus_collection()
        
        st.markdown("---")
        st.caption("Created by Ishan Chakraborty\nMIT License 2024")


def process_uploaded_files(files):
    """Process files with checks for existence."""
    if not files:
        return
    processor = st.session_state.document_processor
    vector_store = st.session_state.vector_store
    progress = st.progress(0)
    status = st.empty()
    total = len(files)
    chunks = 0
    
    for i, file in enumerate(files):
        status.text(f"Processing {file.name}...")
        try:
            # Save and process
            path = processor.save_uploaded_file(file)
            c = processor.process_file(path)
            if c:
                added = vector_store.add_documents(c)
                chunks += added
                update_analytics(chunks_added=added)
        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")
        progress.progress((i + 1) / total)
    
    status.success(f"Successfully processed {chunks} chunks from {len(files)} new file(s).")
    time.sleep(1.5)
    status.empty()
    progress.empty()
    st.rerun()


def delete_file(name):
    """Delete file robustly."""
    try:
        processor = st.session_state.document_processor
        vector_store = st.session_state.vector_store
        
        if vector_store:
            vector_store.delete_by_source(name)
        
        if st.session_state.get("preview_file") and st.session_state.preview_file['name'] == name:
            del st.session_state.preview_file
            
        processor.delete_file(name)
        
        # Update chunk count roughly if possible (re-fetch correct stats)
        if vector_store:
            stats = vector_store.get_collection_stats()
            st.session_state.analytics["total_chunks"] = stats.get("row_count", 0)
            
        st.rerun()
    except Exception as e:
        st.error(f"Error deleting file: {str(e)}")


def clear_all_data():
    """Clear all data and reset state."""
    try:
        processor = st.session_state.document_processor
        vector_store = st.session_state.vector_store
        
        if vector_store:
            vector_store.clear_collection()
        
        files = processor.get_uploaded_files()
        for f in files:
            processor.delete_file(f["name"])
            
        st.session_state.messages = []
        st.session_state.analytics = {
            "total_chunks": 0, "total_tokens_used": 0,
            "total_queries": 0, "estimated_cost": 0.0
        }
        if "preview_file" in st.session_state:
            del st.session_state.preview_file
            
        st.success("System reset: All files deleted and DB emptied.")
        time.sleep(1)
        st.rerun()
    except Exception as e:
        st.error(f"Error clearing data: {str(e)}")


def reset_milvus_collection():
    """Reset Milvus collection only."""
    try:
        if st.session_state.vector_store:
            st.session_state.vector_store.clear_collection()
            
            st.session_state.analytics["total_chunks"] = 0
            
            st.success("Database Flushed: Collection recreated empty.")
            time.sleep(1)
            st.rerun()
    except Exception as e:
        st.error(f"Error resetting DB: {str(e)}")


def render_analytics():
    """Render dashboard with live data."""
    a = st.session_state.analytics
    cols = st.columns(5)
    
    # Calculate documents lively
    doc_count = len(st.session_state.document_processor.get_uploaded_files())
    
    # Chunks are from analytics state (optimistic)
    chunk_count = a["total_chunks"]
    
    metrics = [
        ("Documents", doc_count),
        ("Chunks", chunk_count),
        ("Queries", a["total_queries"]),
        ("Tokens", a["total_tokens_used"]),
        ("Cost", f"${a['estimated_cost']:.4f}")
    ]
    
    for i, (label, value) in enumerate(metrics):
        with cols[i]:
            st.metric(label, value)


def render_chat():
    """Render chat."""
    for msg in st.session_state.messages:
        avatar = "👤" if msg["role"] == "user" else "🤖"
        
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 View Sources"):
                    for src in msg["sources"]:
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>{src['source']}</strong> (Page {src['page']})<br>
                            <small>{src.get('text', '')[:200]}...</small>
                        </div>
                        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.session_state.rephrase_query:
        st.markdown(f"""
        <div class="rephrase-box">
            <div class="rephrase-label">✨ Rephrased Query</div>
            <div class="rephrase-text">{st.session_state.rephrase_query}</div>
        </div>
        """, unsafe_allow_html=True)
        active_query = st.session_state.rephrase_query
    else:
        active_query = ""

    col1, col2 = st.columns([5, 1])
    with col1:
        # Input Area
        query_input = st.text_input(
            "Ask a question", 
            key="query_input", 
            label_visibility="collapsed", 
            placeholder="Ask specific questions or type 'Summarize'..."
        )
    with col2:
        if st.button("Rephrase", use_container_width=True):
            if query_input:
                new_q = rephrase_query_func(query_input)
                st.session_state.rephrase_query = new_q
                st.rerun()
            else:
                st.warning("Enter text first")

    if st.button("Send Query", use_container_width=True, type="primary"):
        input_to_use = active_query if active_query else query_input
        if input_to_use:
            process_query(input_to_use)
            st.session_state.rephrase_query = ""
            st.rerun()


def process_query(prompt):
    """Process query."""
    if not prompt: return
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Thinking..."):
        try:
            crew = st.session_state.crew
            crew.set_model(st.session_state.selected_model)
            res = crew.query(prompt)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": res["answer"],
                "sources": res.get("sources", [])
            })
            
            tok = int((len(prompt.split()) + len(res["answer"].split())) * 1.5)
            update_analytics(tokens_used=tok, query_made=True)
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")


def main():
    init_session_state()
    st.markdown('<h1 class="main-header">📚 Document Knowledge Retrieval</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multi-Agent RAG with Milvus & OpenAI</p>', unsafe_allow_html=True)
    st.markdown('<p class="author-credit">Created by Ishan Chakraborty | MIT License</p>', unsafe_allow_html=True)
    
    if not initialize_services(): st.stop()
    
    render_sidebar()
    
    st.markdown('<div class="analytics-title">📊 Analytics Dashboard</div>', unsafe_allow_html=True)
    render_analytics()
    st.markdown("---")
    
    if st.session_state.get("preview_file"):
        f = st.session_state.preview_file
        st.markdown(f"### 👁️ Preview: {f['name']}")
        render_document_viewer(f['path'], f['name'])
        if st.button("Close Preview"):
            del st.session_state.preview_file
            st.rerun()
        st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📂 Documents", "ℹ️ About"])
    
    with tab1:
        if not st.session_state.document_processor.get_uploaded_files():
            st.info("👋 Welcome! Please upload documents in the sidebar to get started.")
        render_chat()
    
    with tab2:
        st.markdown("### 📂 Document Repository")
        files = st.session_state.document_processor.get_uploaded_files()
        if files:
            sel = st.selectbox("Select document to view", [f['name'] for f in files])
            if sel:
                for f in files:
                    if f['name'] == sel:
                        render_document_viewer(f['path'], f['name'])
        else:
            st.info("No documents available.")
            
    with tab3:
        st.markdown("""
        ### ℹ️ About Document Knowledge Retrieval
        
        This application is an advanced **Retrieval-Augmented Generation (RAG)** system designed to help you extract insights from your documents accurately and efficiently.
        
        #### 🏗️ Architecture
        - **Frontend**: Streamlit (Python)
        - **Brain**: OpenAI GPT-4o / O1 Models
        - **Memory**: Milvus Vector Database (Zilliz Cloud)
        - **Orchestration**: Custom Multi-Agent Crew (LangChain based)
        
        #### ✨ Key Features
        - **Multi-Format Support**: Upload PDF, TXT, and CSV files.
        - **Smart Query Rephrasing**: Takes simple commands like "Summarize" and transforms them into professional prompts using an LLM.
        - **Granular Citations**: Every answer includes specific source citations with page numbers and text snippets.
        - **Cost & Token Tracking**: Monitor your API usage and estimated costs in real-time.
        - **White Label Design**: Professional, clean UI with high contrast and responsiveness.
        
        #### 👨‍💻 Author
        **Created by Ishan Chakraborty**  
        *Full Stack AI Engineer*  
        Licensed under MIT License 2024.
        
        For support or inquiries, please verify the configuration or check the documentation.
        """)

if __name__ == "__main__":
    main()
