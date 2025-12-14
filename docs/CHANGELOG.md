# Changelog

All notable changes to Document Knowledge Retrieval Tool will be documented in this file.

## [1.0.0] - 2024-12-14

### 🎉 Initial Release

This is the first official release of **Document Knowledge Retrieval Tool** - a multi-agent RAG (Retrieval-Augmented Generation) system powered by CrewAI, Milvus, and OpenAI.

🌐 **Live Demo**: https://document-knowledge-retrieval.streamlit.app/

---

### ✨ Features

#### 📄 Document Processing
- **Multi-format support**: Upload and process PDF, TXT, and CSV files
- **Intelligent chunking**: Documents are automatically split using `RecursiveCharacterTextSplitter` with configurable chunk size (default: 1000) and overlap (default: 200)
- **Metadata preservation**: Source file names, page numbers, and chunk indices are tracked

#### 🔍 Vector Search & Retrieval
- **Milvus/Zilliz Cloud integration**: Serverless vector database for scalable document storage
- **OpenAI embeddings**: Uses `text-embedding-3-large` (3072 dimensions) for high-accuracy semantic search
- **Cosine similarity**: Precise relevance scoring for document retrieval

#### 🤖 Multi-Agent System
- **Retrieval Agent**: Specializes in finding and prioritizing relevant document chunks
- **Response Agent**: Synthesizes retrieved information into clear, comprehensive answers
- **Analyzer Agent**: Provides document analysis and insights

#### 🎨 Streamlit UI
- **Clean, modern interface** with custom CSS styling
- **Real-time analytics dashboard**: Track token usage, chunk counts, and estimated costs
- **Interactive chat interface** with message history
- **Query rephrasing**: AI-powered query optimization for better retrieval
- **Source citations**: Expandable view showing exact text snippets used in responses
- **Multiple model support**: GPT-4o, GPT-4o Mini, GPT-4 Turbo, o1-preview, o1-mini

#### ⚙️ Configuration
- **Environment-based config**: Easy setup via `.env` file
- **Configurable parameters**: Model selection, chunk size, embedding model
- **Validation utilities**: Built-in config validation

---

### 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit 1.28+ |
| LLM | OpenAI GPT-4o / GPT-4o Mini |
| Embeddings | OpenAI text-embedding-3-large |
| Vector DB | Milvus / Zilliz Cloud |
| Orchestration | Custom CrewAI-style agents |
| Document Loading | LangChain (PyPDF, TextLoader, CSVLoader) |

---

### 📦 Dependencies

```
streamlit>=1.28.0
langchain>=0.1.0
langchain-community>=0.0.20
langchain-openai>=0.0.5
langchain-text-splitters>=0.0.1
langchain-core>=0.1.0
pypdf>=3.17.0
pymilvus>=2.3.0
openai>=1.0.0
python-dotenv>=1.0.0
```

---

### 📁 Project Structure

```
├── app.py                    # Streamlit entry point
├── config.py                 # Configuration management
├── requirements.txt          # Dependencies
├── src/
│   ├── document_processor.py # PDF/document handling
│   ├── vector_store.py       # Milvus integration
│   ├── agents.py             # CrewAI-style agents
│   ├── tasks.py              # Agent tasks
│   └── crew.py               # Crew orchestration
└── docs/
    └── deployment.md         # Deployment guide
```

---

### 🚀 Deployment

- **Streamlit Community Cloud** deployment ready
- Comprehensive [deployment guide](docs/deployment.md) included
- TOML-based secrets configuration support

---

### 👤 Author

**Ishan Chakraborty**

---

### 📄 License

MIT License

---

**Full Changelog**: https://github.com/Ishan96Dev/document-knowledge-retrieval/commits/v1.0
