import os
import logging
from typing import List, Optional
from tqdm import tqdm
import gc
import torch
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
BATCH_SIZE = 64  # For processing documents in batches

def load_pdf_files(data_path: str) -> List[Document]:
    """Load PDF files with error handling and progress tracking"""
    try:
        logger.info(f"Loading PDF files from {data_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Directory not found: {data_path}")

        loader = DirectoryLoader(
            data_path,
            glob='*.pdf',
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        
        documents = loader.load()
        logger.info(f"Successfully loaded {len(documents)} pages from PDF files")
        return documents
    except Exception as e:
        logger.error(f"Error loading PDF files: {str(e)}")
        raise

def create_chunks(extracted_data: List[Document], chunk_size: int = CHUNK_SIZE, 
                 chunk_overlap: int = CHUNK_OVERLAP) -> List[Document]:
    """Create text chunks with memory-efficient processing"""
    try:
        logger.info("Creating text chunks")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        # Process in batches to manage memory
        chunks = []
        for i in range(0, len(extracted_data), BATCH_SIZE):
            batch = extracted_data[i:i + BATCH_SIZE]
            batch_chunks = text_splitter.split_documents(batch)
            chunks.extend(batch_chunks)
            
            # Force garbage collection after each batch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info(f"Created {len(chunks)} text chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error creating chunks: {str(e)}")
        raise

def get_embedding_model():
    """Initialize embedding model with error handling"""
    try:
        logger.info("Initializing embedding model")
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        return embedding_model
    except Exception as e:
        logger.error(f"Error initializing embedding model: {str(e)}")
        raise

def create_and_save_vectorstore(chunks: List[Document], embedding_model, 
                              save_path: str = DB_FAISS_PATH):
    """Create and save vector store with progress tracking and error handling"""
    try:
        logger.info("Creating FAISS vector store")
        
        # Ensure the save directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Process in batches for memory efficiency
        db = None
        for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Creating embeddings"):
            batch = chunks[i:i + BATCH_SIZE]
            if db is None:
                db = FAISS.from_documents(batch, embedding_model)
            else:
                batch_db = FAISS.from_documents(batch, embedding_model)
                db.merge_from(batch_db)
            
            # Clean up memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info(f"Saving vector store to {save_path}")
        db.save_local(save_path)
        logger.info("Vector store creation completed successfully")
        return db
    except Exception as e:
        logger.error(f"Error creating/saving vector store: {str(e)}")
        raise

def main():
    try:
        # Load documents
        documents = load_pdf_files(DATA_PATH)
        
        # Create text chunks
        text_chunks = create_chunks(documents)
        
        # Initialize embedding model
        embedding_model = get_embedding_model()
        
        # Create and save vector store
        create_and_save_vectorstore(text_chunks, embedding_model)
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()