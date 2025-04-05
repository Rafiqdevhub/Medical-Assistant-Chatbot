import os
import logging
import streamlit as st
from typing import Optional
from datetime import datetime
import time
from functools import lru_cache
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DB_FAISS_PATH = "vectorstore/db_faiss"
MAX_RETRIES = 3
RETRY_DELAY = 1
REQUEST_TIMEOUT = 30
MAX_REQUESTS_PER_MINUTE = 30

# Rate limiting
request_timestamps = []

def rate_limit():
    """Implement rate limiting"""
    current_time = time.time()
    request_timestamps.append(current_time)
    
    # Remove timestamps older than 1 minute
    while request_timestamps and request_timestamps[0] < current_time - 60:
        request_timestamps.pop(0)
    
    if len(request_timestamps) > MAX_REQUESTS_PER_MINUTE:
        raise Exception("Rate limit exceeded. Please try again later.")

@st.cache_resource
def get_vectorstore() -> Optional[FAISS]:
    """Load and cache the vector store with proper error handling"""
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            cache_folder='./model_cache'
        )
        db = FAISS.load_local(
            DB_FAISS_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        return db
    except Exception as e:
        logger.error(f"Failed to load vector store: {str(e)}")
        return None

def set_custom_prompt(custom_prompt_template: str) -> PromptTemplate:
    """Create a prompt template with input validation"""
    if not isinstance(custom_prompt_template, str):
        raise ValueError("Prompt template must be a string")
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )

@lru_cache(maxsize=1)
def load_llm(huggingface_repo_id: str, HF_TOKEN: str):
    """Load and cache the language model with retry logic"""
    for attempt in range(MAX_RETRIES):
        try:
            return HuggingFaceEndpoint(
                repo_id=huggingface_repo_id,
                temperature=0.5,
                model_kwargs={
                    "token": HF_TOKEN,
                    "max_length": "512",
                    "timeout": REQUEST_TIMEOUT
                }
            )
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            logger.warning(f"Attempt {attempt + 1} failed, retrying... Error: {str(e)}")
            time.sleep(RETRY_DELAY * (attempt + 1))

def validate_input(prompt: str) -> bool:
    """Validate user input"""
    if not prompt or len(prompt.strip()) == 0:
        return False
    if len(prompt) > 500:  # Reasonable limit for input length
        return False
    return True

def format_source_documents(source_docs) -> str:
    """Format source documents in a clean, readable way"""
    formatted_sources = []
    for i, doc in enumerate(source_docs, 1):
        # Extract relevant metadata and content
        content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        formatted_sources.append(f"Source {i}:\n{content}\n")
    return "\n".join(formatted_sources)

def main():
    st.set_page_config(
        page_title="Medical Chatbot",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 Medical Assistant Chatbot")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'error_count' not in st.session_state:
        st.session_state.error_count = 0
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # Chat input
    prompt = st.chat_input("What would you like to know about medical conditions?")

    if prompt:
        try:
            # Rate limiting check
            rate_limit()
            
            # Input validation
            if not validate_input(prompt):
                st.error("Please provide a valid question (between 1 and 500 characters)")
                return
            
            # Display user message
            with st.chat_message('user'):
                st.markdown(prompt)
            st.session_state.messages.append({'role': 'user', 'content': prompt})

            # Show processing indicator
            with st.spinner("Thinking..."):
                CUSTOM_PROMPT_TEMPLATE = """
                Use the following context to answer the user's medical question.
                If you don't know the answer or aren't completely sure, say so - do not make up information.
                Only provide information that is supported by the given context.
                If the question is not medical-related, kindly redirect the user to ask medical questions.

                Context: {context}
                Question: {question}

                Answer the question directly and professionally.
                """
                
                HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
                HF_TOKEN = os.environ.get("HF_TOKEN")
                
                if not HF_TOKEN:
                    raise ValueError("HF_TOKEN environment variable is not set")

                vectorstore = get_vectorstore()
                if vectorstore is None:
                    raise RuntimeError("Failed to initialize vector store")

                qa_chain = RetrievalQA.from_chain_type(
                    llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                )

                response = qa_chain.invoke({'query': prompt})
                
                # Format response
                result = response["result"]
                source_documents = format_source_documents(response["source_documents"])
                
                result_to_show = (
                    f"{result}\n\n"
                    f"---\n"
                    f"📚 **Reference Sources:**\n{source_documents}"
                )

                # Display assistant response
                with st.chat_message('assistant'):
                    st.markdown(result_to_show)
                st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})
                
                # Reset error count on successful response
                st.session_state.error_count = 0

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            st.session_state.error_count += 1
            
            error_message = "I apologize, but I encountered an error processing your request."
            if st.session_state.error_count >= 3:
                error_message += "\nIt seems there might be persistent issues. Please try again later or contact support."
            
            with st.chat_message('assistant'):
                st.error(error_message)
            st.session_state.messages.append({'role': 'assistant', 'content': error_message})

if __name__ == "__main__":
    main()