import os
import logging
from typing import Dict, Any, Optional
from functools import lru_cache
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.callbacks import get_openai_callback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DB_FAISS_PATH = "vectorstore/db_faiss"
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_RETRIES = 3
RETRY_DELAY = 1
REQUEST_TIMEOUT = 30

class ModelConnectionError(Exception):
    """Custom exception for model connection issues"""
    pass

@lru_cache(maxsize=1)
def get_hf_token() -> str:
    """Get and validate HuggingFace token"""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable is not set")
    return token

@lru_cache(maxsize=1)
def load_llm(huggingface_repo_id: str) -> HuggingFaceEndpoint:
    """Load LLM with caching and retry logic"""
    for attempt in range(MAX_RETRIES):
        try:
            llm = HuggingFaceEndpoint(
                repo_id=huggingface_repo_id,
                temperature=0.5,
                model_kwargs={
                    "token": get_hf_token(),
                    "max_length": "512",
                    "timeout": REQUEST_TIMEOUT
                }
            )
            return llm
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise ModelConnectionError(f"Failed to load LLM after {MAX_RETRIES} attempts: {str(e)}")
            logger.warning(f"Attempt {attempt + 1} failed, retrying... Error: {str(e)}")
            time.sleep(RETRY_DELAY * (attempt + 1))

def set_custom_prompt() -> PromptTemplate:
    """Create an optimized prompt template"""
    template = """
    Use the following context to answer the user's medical question professionally and accurately.
    If you don't know the answer or aren't completely sure, say so - do not make up information.
    Only provide information that is supported by the given context.
    If the question is not medical-related, kindly redirect the user to ask medical questions.

    Context: {context}
    Question: {question}

    Answer the question directly and professionally.
    If the condition is serious, always recommend consulting a healthcare professional.
    """
    return PromptTemplate(
        template=template.strip(),
        input_variables=["context", "question"]
    )

@lru_cache(maxsize=1)
def load_vectorstore() -> FAISS:
    """Load vector store with caching and validation"""
    try:
        if not Path(DB_FAISS_PATH).exists():
            raise FileNotFoundError(f"Vector store not found at {DB_FAISS_PATH}")
        
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder='./model_cache'
        )
        
        db = FAISS.load_local(
            DB_FAISS_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        return db
    except Exception as e:
        raise ModelConnectionError(f"Failed to load vector store: {str(e)}")

def create_qa_chain() -> RetrievalQA:
    """Create optimized QA chain"""
    try:
        db = load_vectorstore()
        qa_chain = RetrievalQA.from_chain_type(
            llm=load_llm(HUGGINGFACE_REPO_ID),
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': set_custom_prompt()}
        )
        return qa_chain
    except Exception as e:
        raise ModelConnectionError(f"Failed to create QA chain: {str(e)}")

def process_query(query: str) -> Dict[str, Any]:
    """Process a user query with error handling and validation"""
    try:
        # Input validation
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if len(query) > 500:
            raise ValueError("Query is too long (max 500 characters)")

        # Create QA chain if needed
        qa_chain = create_qa_chain()
        
        # Process query with timing
        start_time = time.time()
        response = qa_chain.invoke({'query': query})
        processing_time = time.time() - start_time
        
        logger.info(f"Query processed in {processing_time:.2f} seconds")
        
        return {
            'result': response["result"],
            'source_documents': response["source_documents"],
            'processing_time': processing_time
        }
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise

def main():
    """Main interaction loop with improved error handling"""
    try:
        print("Medical Chatbot initialized. Type 'quit' to exit.")
        while True:
            user_query = input("\nWrite Query Here: ").strip()
            
            if user_query.lower() in ('quit', 'exit'):
                break
                
            try:
                response = process_query(user_query)
                print("\nRESULT:", response["result"])
                print("\nProcessing Time:", f"{response['processing_time']:.2f} seconds")
                
                # Only show source documents if they exist
                if response.get("source_documents"):
                    print("\nSOURCE DOCUMENTS:", response["source_documents"])
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
                print("Please try again with a different query.")
                
    except KeyboardInterrupt:
        print("\nChatbot terminated by user.")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        print("An unexpected error occurred. Please check the logs for details.")

if __name__ == "__main__":
    main()