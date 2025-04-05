import os
import logging
import streamlit as st
from typing import Optional
from datetime import datetime
import time
from functools import lru_cache
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

load_dotenv()
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

CUSTOM_CSS = """
<style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Header styling */
    .stApp header {
        background-color: #2c3e50;
        color: white;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left: 5px solid #1976d2;
    }
    
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 5px solid #43a047;
    }
    
    /* Medical dashboard styling */
    .dashboard-metric {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin: 0.5rem 0;
    }
    
    /* Custom button styling */
    .stButton>button {
        background-color: #2196f3;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Sources section styling */
    .sources-section {
        background-color: #fff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #ff9800;
        margin-top: 1rem;
    }
    
    /* Medical disclaimer styling */
    .medical-disclaimer {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ffcdd2;
        margin: 1rem 0;
    }
    
    /* Engagement features styling */
    .feedback-section {
        margin-top: 1rem;
        padding: 1rem;
        background-color: #e8f5e9;
        border-radius: 8px;
    }
</style>
"""

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
                task="text-generation",
                temperature=0.5,
                timeout=REQUEST_TIMEOUT,
                model_kwargs={
                    "token": HF_TOKEN,
                    "max_length": "512"
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
        # Extract page number if available
        page_num = doc.metadata.get('page', 'N/A')
        # Extract relevant metadata and content
        content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        formatted_sources.append(f"📄 **Source {i}** (Page {page_num}):\n>{content}\n")
    return "\n".join(formatted_sources)

def display_welcome():
    """Display welcome message and doctor's introduction"""
    if 'first_visit' not in st.session_state:
        st.session_state.first_visit = False
        st.balloons()
        
        with st.expander("👨‍⚕️ Welcome to Your Virtual Medical Consultation", expanded=True):
            col1, col2 = st.columns([1, 3])
            with col2:
                st.markdown("""
                Hello! I'm your AI Medical Assistant. While I'm not a replacement for your doctor, 
                I'm here to provide you with reliable medical information and guidance.

                **How I can help you:**
                - Answer medical questions with evidence-based information
                - Explain medical terms and conditions
                - Provide general health information
                - Help you understand symptoms
                
                **Remember:** Always consult with healthcare professionals for personal medical advice.
                """)
            
            st.markdown("""
            ### 🎯 Tips for the best consultation:
            1. **Be Specific:** Instead of "I have pain," try "I have sharp knee pain when climbing stairs"
            2. **One Topic:** Focus on one health topic at a time for detailed information
            3. **Ask Follow-ups:** Don't hesitate to ask for clarification
            4. **Share Context:** Include relevant details about your query
            
            ### 📝 Example Questions:
            - "What are the early warning signs of type 2 diabetes?"
            - "Can you explain what causes seasonal allergies?"
            - "What are common treatments for migraine headaches?"
            - "What lifestyle changes help prevent heart disease?"
            """)

def show_user_dashboard():
    """Display user consultation dashboard"""
    st.sidebar.markdown("### 📊 Consultation Dashboard")
    
    # Session metrics
    with st.sidebar.container():
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Questions Asked", st.session_state.get('total_questions', 0))
        with col2:
            remaining_requests = MAX_REQUESTS_PER_MINUTE - len(request_timestamps)
            st.metric("Available Queries", remaining_requests)
    
    # Consultation timer
    if 'start_time' not in st.session_state:
        st.session_state.start_time = time.time()
    
    consultation_time = int(time.time() - st.session_state.start_time)
    st.sidebar.markdown(f"⏱️ Session Duration: {consultation_time // 60}m {consultation_time % 60}s")
    
    # Health resources
    st.sidebar.markdown("### 🏥 Health Resources")
    st.sidebar.markdown("""
    - [CDC Health Information](https://www.cdc.gov)
    - [WHO Health Topics](https://www.who.int/health-topics)
    - [MedlinePlus](https://medlineplus.gov)
    - [Find a Doctor](https://www.healthgrades.com)
    """)
    
    # Medical disclaimer
    st.sidebar.markdown("### ⚕️ Medical Disclaimer")
    st.sidebar.info(
        "This AI assistant provides general medical information only. "
        "It is not a substitute for professional medical advice, diagnosis, or treatment. "
        "Always seek the advice of your physician or other qualified health provider."
    )

def show_feedback_section(response):
    """Display feedback section for user engagement"""
    st.markdown("### 💭 Was this information helpful?")
    col1, col2, col3 = st.columns([1,1,2])
    with col1:
        if st.button("👍 Yes"):
            st.success("Thank you for your feedback! We're glad we could help.")
    with col2:
        if st.button("👎 No"):
            st.warning("We're sorry we couldn't help. Please try rephrasing your question or consult a healthcare professional.")
    with col3:
        st.markdown("_Your feedback helps us improve!_")

def format_response(result, source_documents):
    """Format the response in a professional medical style"""
    st.markdown("### 📋 Medical Information")
    st.markdown(result)
    
    if source_documents:
        st.markdown("### 📚 Medical Sources")
        st.markdown("Click on each source to view details:")
        
        for i, doc in enumerate(source_documents, 1):
            with st.container():
                if st.button(f"Source {i} 📄", key=f"source_{i}"):
                    st.markdown(f"""
                    ```
                    {doc.page_content[:200]}...
                    ```
                    """)
    
    st.markdown("---")
    st.markdown("""
    🏥 **Remember:** This information is for educational purposes only. 
    Consult with healthcare professionals for medical advice specific to your situation.
    """)

def show_error(message: str, level: str = "error"):
    """Show error message with appropriate styling"""
    with st.chat_message('assistant', avatar="⚠️"):
        if level == "warning":
            st.warning(message)
        else:
            st.error(message)
        
        if level == "error":
            st.button("Clear Chat", on_click=lambda: st.session_state.clear())

def main():
    st.set_page_config(
        page_title="Virtual Medical Consultation",
        page_icon="👨‍⚕️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Display header
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.title("👨‍⚕️ Virtual Medical Consultation")
        st.markdown("""
        Your trusted source for medical information and guidance.
        Please ask your health-related questions below.
        """)
    
    # Show user dashboard
    show_user_dashboard()
    
    # Display welcome message for first-time users
    display_welcome()
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'error_count' not in st.session_state:
        st.session_state.error_count = 0
    
    # Chat interface
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message['role'], avatar="👨‍⚕️" if message['role'] == 'assistant' else "🤵"):
                st.markdown(message['content'])
    
    # User input
    prompt = st.chat_input("Type your medical question here... (e.g., 'What are the symptoms of the flu?')")

    if prompt:
        try:
            # Show typing indicator
            with st.chat_message('assistant', avatar="👨‍⚕️"):
                with st.empty():
                    st.markdown("_Consulting medical sources..._")
                    time.sleep(0.5)
            
            # Rate limiting check
            try:
                rate_limit()
            except Exception as e:
                show_error("⏳ " + str(e), level="warning")
                return
            
            # Input validation
            if not validate_input(prompt):
                show_error("⚠️ Please provide a valid question (between 1 and 500 characters)")
                return
            
            # Display user message
            with st.chat_message('user'):
                st.markdown(prompt)
            st.session_state.messages.append({'role': 'user', 'content': prompt})

            # Process the query with improved visual feedback
            with st.status("🔍 Processing your question...", expanded=True) as status:
                st.write("Searching medical encyclopedia...")
                
                CUSTOM_PROMPT_TEMPLATE = """
                Use the following context to answer the user's medical question.
                If you don't know the answer or aren't completely sure, say so - do not make up information.
                Only provide information that is supported by the given context.
                If the question is not medical-related, kindly redirect the user to ask medical questions.

                Context: {context}
                Question: {question}

                Answer the question directly and professionally.
                """
                
                # Update status
                status.write("Analyzing relevant medical information...")
                
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
                format_response(result, response["source_documents"])
                st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})
                
                # Reset error count on successful response
                st.session_state.error_count = 0
                
                # Update final status
                status.update(label="✅ Response ready!", state="complete", expanded=False)

                # Show feedback section
                show_feedback_section(result)

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            st.session_state.error_count += 1
            
            error_message = (
                "🏥 I apologize, but I encountered an error processing your request.\n\n"
                f"Error details: _{str(e)}_"
            )
            
            if st.session_state.error_count >= 3:
                error_message += (
                    "\n\n❗ It seems there might be persistent issues. "
                    "Please try:\n"
                    "1. Refreshing the page\n"
                    "2. Clearing your browser cache\n"
                    "3. Trying again in a few minutes\n"
                    "\nIf the problem persists, please contact support."
                )
            
            show_error(error_message)
            st.session_state.messages.append({'role': 'assistant', 'content': error_message})

if __name__ == "__main__":
    main()