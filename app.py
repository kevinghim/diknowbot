import streamlit as st
from streamlit_chat import message
import os
import tempfile
from typing import List, Dict, Any, Optional

# Import custom modules
from loaders.notion_loader import NotionLoader
from loaders.pdf_loader import PDFLoader
from loaders.docx_loader import DocxLoader
from utils.text_processing import process_documents
from utils.vector_store import (
    connect_to_vectorstore,
    load_data_into_vectorstore,
    load_chain
)

# Set page config
st.set_page_config(
    page_title="Document Chat Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []
    
if 'documents_loaded' not in st.session_state:
    st.session_state['documents_loaded'] = False
    
if 'temp_file_paths' not in st.session_state:
    st.session_state['temp_file_paths'] = []

if 'submit_pressed' not in st.session_state:
    st.session_state.submit_pressed = False

# Function to save uploaded files temporarily
def save_uploaded_file(uploaded_file) -> str:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            st.session_state['temp_file_paths'].append(tmp_file.name)
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return ""

# Function to clean up temporary files
def cleanup_temp_files():
    for file_path in st.session_state['temp_file_paths']:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            st.error(f"Error removing temporary file {file_path}: {str(e)}")
    st.session_state['temp_file_paths'] = []

def handle_enter(key):
    if key == 'Enter':
        st.session_state.submit_pressed = True

# Main app title
st.title('📚 Chat with Your Documents')
st.subheader('Ask questions about your Notion, PDF, and Word documents')

# Set up sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # API Keys
    api_key_tab, file_upload_tab, model_config_tab = st.tabs(["API Keys", "Upload Files", "Model Config"])
    
    with api_key_tab:
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password", 
            help="Enter your OpenAI API key",
            placeholder="sk-...",
        )
        
        anthropic_api_key = st.text_input(
            "Anthropic API Key (Optional)",
            type="password",
            help="Enter your Anthropic API key if using Claude",
            placeholder="sk-ant-...",
        )
        
        notion_api_key = st.text_input(
            "Notion API Key (Optional)",
            type="password",
            help="Enter your Notion API key to load from Notion",
            placeholder="secret_...",
        )
    
    with file_upload_tab:
        st.subheader("Upload Documents")
        
        # File uploader for PDFs
        pdf_files = st.file_uploader(
            "Upload PDF documents", 
            type=["pdf"], 
            accept_multiple_files=True
        )
        
        # File uploader for Word documents
        docx_files = st.file_uploader(
            "Upload Word documents", 
            type=["docx"], 
            accept_multiple_files=True
        )
        
        # Notion integration
        st.subheader("Notion Integration")
        load_from_notion = st.checkbox(
            "Load documents from Notion",
            help="Requires Notion API Key"
        )
        
    with model_config_tab:
        st.subheader("Model Configuration")
        
        model_provider = st.selectbox(
            "Choose Model Provider",
            options=["OpenAI", "Anthropic"],
            index=0,
        )
        
        if model_provider == "OpenAI":
            model_name = st.selectbox(
                "Select OpenAI Model",
                options=["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4"],
                index=0,
            )
        else:
            model_name = st.selectbox(
                "Select Anthropic Model",
                options=["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
                index=1,
            )
            
        # Vector store settings
        st.subheader("Vector Database Settings")
        
        # Check if running on Streamlit Cloud
        is_deployed = os.environ.get('STREAMLIT_DEPLOYED', False)

        # Initialize qdrant settings
        if is_deployed:
            qdrant_host = "https://6037f3a0-f569-4322-bbfa-179e30253d9d.us-east4-0.gcp.cloud.qdrant.io"
            qdrant_api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.40v8NLDkLdBIKpVjWJuU3ByUr8uVCZ5lpdVcm7q6I3A"
            qdrant_port = None
            
            # Hide the settings in deployed version
            st.write("Using cloud Qdrant instance")
        else:
            # Local development settings
            qdrant_host = st.text_input(
                "Qdrant Host",
                value="localhost",
                help="Hostname for Qdrant vector database"
            )
            qdrant_port = st.number_input(
                "Qdrant Port",
                value=6333,
                help="Port for Qdrant vector database"
            )
            qdrant_api_key = None

        collection_name = st.text_input(
            "Collection Name",
            value="documents_collection",
            help="Name for the vector collection"
        )
    
    # Load data button
    load_data_button = st.button("Load Documents into Database", use_container_width=True)

# Document Processing Logic
if load_data_button:
    if not openai_api_key:
        st.sidebar.error("⚠️ Please provide an OpenAI API key")
    else:
        with st.spinner("Loading and processing documents..."):
            try:
                # Initialize vector store
                vector_store = connect_to_vectorstore(
                    host=qdrant_host,
                    port=qdrant_port,
                    api_key=qdrant_api_key,
                    collection_name=collection_name
                )
                
                all_documents = []
                
                # Process Notion documents if requested
                if load_from_notion and notion_api_key:
                    try:
                        notion_loader = NotionLoader(notion_api_key)
                        notion_docs = notion_loader.load_documents()
                        if notion_docs:
                            st.sidebar.success(f"✅ Loaded {len(notion_docs)} documents from Notion")
                            all_documents.extend(notion_docs)
                        else:
                            st.sidebar.warning("No documents found in Notion")
                    except Exception as e:
                        st.sidebar.error(f"Error loading from Notion: {str(e)}")
                
                # Process PDF files
                if pdf_files:
                    pdf_loader = PDFLoader(ocr_enabled=True)
                    for pdf_file in pdf_files:
                        try:
                            temp_path = save_uploaded_file(pdf_file)
                            if temp_path:
                                pdf_doc = pdf_loader.load_document(temp_path)
                                if pdf_doc:
                                    all_documents.append(pdf_doc['content'])
                                    st.sidebar.success(f"✅ Loaded PDF: {pdf_file.name}")
                        except Exception as e:
                            st.sidebar.error(f"Error processing PDF {pdf_file.name}: {str(e)}")
                
                # Process Word documents
                if docx_files:
                    docx_loader = DocxLoader()
                    for docx_file in docx_files:
                        try:
                            temp_path = save_uploaded_file(docx_file)
                            if temp_path:
                                docx_doc = docx_loader.load_document(temp_path)
                                if docx_doc:
                                    all_documents.append(docx_doc['content'])
                                    st.sidebar.success(f"✅ Loaded Word document: {docx_file.name}")
                        except Exception as e:
                            st.sidebar.error(f"Error processing Word document {docx_file.name}: {str(e)}")
                
                # Process and load documents into vector store
                if all_documents:
                    # Process documents into chunks
                    chunks = process_documents(all_documents)
                    
                    # Load chunks into vector store
                    load_data_into_vectorstore(
                        vector_store,
                        chunks,
                        openai_api_key,
                        collection_name
                    )
                    
                    st.session_state['documents_loaded'] = True
                    st.sidebar.success("✅ All documents loaded and processed successfully!")
                else:
                    st.sidebar.warning("No documents were loaded. Please upload documents or enable Notion integration.")
                
                # Clean up temporary files
                cleanup_temp_files()
                
            except Exception as e:
                st.error(f"Error during document processing: {str(e)}")
                cleanup_temp_files()

# Chat interface
if st.session_state['documents_loaded']:
    try:
        # Initialize the chain
        vector_store = connect_to_vectorstore(
            host=qdrant_host,
            port=qdrant_port,
            collection_name=collection_name
        )
        
        api_key = anthropic_api_key if model_provider == "Anthropic" else openai_api_key
        chain = load_chain(
            vector_store,
            api_key,
            collection_name,
            model_provider.lower(),
            model_name
        )
        
        # Display chat history (newest first, growing upward)
        if st.session_state['generated']:
            for i in range(len(st.session_state['generated']) - 1, -1, -1):
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i][1], key=str(i))
        
        # Empty space to push input to bottom
        st.empty()
        
        # Chat input form at bottom
        with st.form(key="chat_form", clear_on_submit=True):
            col1, col2 = st.columns([0.85, 0.15])
            with col1:
                user_input = st.text_area(
                    "Ask a question about your documents:",
                    height=None,
                    key="input_area"
                )
            with col2:
                submit_button = st.form_submit_button("↑")

            if submit_button and user_input:
                try:
                    question = user_input.strip()
                    # Get response from chain
                    result = chain({
                        "question": question, 
                        "chat_history": st.session_state["generated"]
                    })
                    response = result['answer']
                    
                    # Update chat history
                    st.session_state['past'].append(question)
                    st.session_state['generated'].append((question, response))
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                
    except Exception as e:
        st.error(f"Error initializing chat interface: {str(e)}")
else:
    st.info("👆 Please load your documents using the sidebar to start chatting!")

# Add a footer
st.markdown("---")
st.markdown(
    "Made with ❤️ using Streamlit, LangChain, and Qdrant. "
    "Upload your documents and start chatting!"
)