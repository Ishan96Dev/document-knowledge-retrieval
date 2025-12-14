"""
Document processing module for loading, chunking, and extracting text from documents.

============================================================
Created by: Ishan Chakraborty
License: MIT License
Copyright (c) 2024 Ishan Chakraborty
============================================================
"""
import os
from pathlib import Path
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_core.documents import Document

import config



class DocumentProcessor:
    """Handles document loading, chunking, and text extraction."""
    
    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.upload_dir = Path("uploaded_files")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load a document based on its file extension."""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif extension == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif extension == ".csv":
            loader = CSVLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}")
        
        documents = loader.load()
        
        # Add source metadata
        for doc in documents:
            doc.metadata["source"] = path.name
            doc.metadata["file_path"] = str(path)
        
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks."""
        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk index to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
        
        return chunks
    
    def process_file(self, file_path: str) -> List[Document]:
        """Load and chunk a single file."""
        documents = self.load_document(file_path)
        chunks = self.chunk_documents(documents)
        return chunks
    
    def process_directory(self, directory_path: str) -> List[Document]:
        """Process all supported files in a directory."""
        all_chunks = []
        path = Path(directory_path)
        
        for extension in config.SUPPORTED_EXTENSIONS:
            for file_path in path.glob(f"*{extension}"):
                try:
                    chunks = self.process_file(str(file_path))
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        return all_chunks
    
    def save_uploaded_file(self, uploaded_file) -> str:
        """Save an uploaded file to the uploads directory and return the path."""
        file_path = config.UPLOADS_DIR / uploaded_file.name
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return str(file_path)
    
    def get_uploaded_files(self) -> List[Dict[str, Any]]:
        """Get list of all uploaded files with metadata."""
        files = []
        
        for file_path in config.UPLOADS_DIR.iterdir():
            if file_path.suffix.lower() in config.SUPPORTED_EXTENSIONS:
                files.append({
                    "name": file_path.name,
                    "path": str(file_path),
                    "size": file_path.stat().st_size,
                    "extension": file_path.suffix.lower()
                })
        
        return files
    
    def delete_file(self, file_name: str) -> bool:
        """Delete a file from the uploads directory."""
        file_path = config.UPLOADS_DIR / file_name
        
        if file_path.exists():
            os.remove(file_path)
            return True
        
        return False
