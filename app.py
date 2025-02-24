import streamlit as st
from streamlit_chat import message
import os
import tempfile
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

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
    page_title="Dino the Knowledge Bot",
    page_icon="assets/dino_icon.png",
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
col1, col2 = st.columns([1, 10])  # Adjust ratio as needed
with col1:
    st.image("assets/dino_icon.png", width=60)
with col2:
    st.title('Dino the Knowledge Bot')
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
        try:
            is_deployed = st.secrets["STREAMLIT_DEPLOYED"]
            qdrant_host = st.secrets["QDRANT_HOST"]
            qdrant_api_key = st.secrets["QDRANT_API_KEY"]
            qdrant_port = None
            st.write(f"Using cloud Qdrant instance at {qdrant_host}")
        except Exception:
            # Local development settings
            is_deployed = False
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
    
    # Debug output
    st.write(f"Debug - is_deployed: {is_deployed}")
    st.write(f"Debug - qdrant_host: {qdrant_host}")

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
                vector_store, embeddings = connect_to_vectorstore(
                    host=qdrant_host,
                    port=qdrant_port,
                    api_key=qdrant_api_key,
                    collection_name=collection_name,
                    openai_api_key=openai_api_key
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
                
                if all_documents:
                    try:
                        # Process documents into chunks
                        chunks = process_documents(all_documents)
                        st.write(f"Debug - Created {len(chunks)} text chunks")
                        
                        # Load chunks into vector store
                        load_data_into_vectorstore(
                            vector_store,
                            chunks,
                            openai_api_key,
                            collection_name,
                            {
                                'is_cloud': is_deployed,
                                'host': qdrant_host,
                                'port': qdrant_port,
                                'api_key': qdrant_api_key
                            }
                        )
                        
                        # Store the client and embeddings in session state
                        st.session_state['vector_store'] = vector_store
                        st.session_state['embeddings'] = embeddings
                        st.session_state['documents_loaded'] = True
                        st.sidebar.success("✅ All documents loaded and processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
                else:
                    st.sidebar.warning("No documents were loaded. Please upload documents or enable Notion integration.")
                
                # Clean up temporary files
                cleanup_temp_files()
                
                # Make sure these environment variables are set for the rest of the app
                os.environ['QDRANT_HOST'] = qdrant_host
                os.environ['QDRANT_API_KEY'] = qdrant_api_key if qdrant_api_key else ''
                
            except Exception as e:
                st.error(f"Error during document processing: {str(e)}")
                cleanup_temp_files()

# Chat interface
if st.session_state['documents_loaded']:
    try:
        client = st.session_state['vector_store']
        embeddings = st.session_state['embeddings']
        
        # Create Qdrant wrapper
        vectorstore = Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=embeddings
        )
        
        # Test search with Document objects
        docs = vectorstore.similarity_search("test", k=1)
        if not docs:
            st.error("No documents found in search")
            st.stop()
            
        st.write(f"Found document: {docs[0].page_content[:100]}")
        
        # Create the chat model
        if model_provider == "Anthropic":
            llm = ChatAnthropic(
                model=model_name,
                anthropic_api_key=anthropic_api_key,
                temperature=0.7
            )
        else:
            llm = ChatOpenAI(
                model=model_name,
                openai_api_key=openai_api_key,
                temperature=0.7
            )
        
        # Create the conversation chain
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=memory
        )
        
        # Chat interface
        if 'submit_pressed' not in st.session_state:
            st.session_state.submit_pressed = False
            
        user_input = st.text_input("Ask a question about your documents:", key="input")
        
        if user_input and not st.session_state.submit_pressed:
            st.session_state.submit_pressed = True
            with st.spinner("Thinking..."):
                try:
                    result = qa_chain({"question": user_input})
                    st.session_state['past'].append(user_input)
                    st.session_state['generated'].append(result['answer'])
                except Exception as e:
                    st.error(str(e))
        
        # Display chat history
        if st.session_state['generated']:
            for i in range(len(st.session_state['generated']) - 1, -1, -1):
                message(
                    st.session_state['past'][i],
                    is_user=True,
                    avatar_style="avataaars",  # Options: "initials", "bottts", "avataaars", "jdenticon", "gridy", "identicon", "micah", "miniavs", "adventurer", "big-ears", "big-smile", "bottts", "croodles", "fun-emoji", "icons", "lorelei", "notionists", "open-peeps", "personas", "pixel-art", "thumbs"
                    key=str(i) + '_user'
                )
                message(
                    st.session_state["generated"][i],
                    avatar_style="bottts",  # Different style for the bot
                    key=str(i)
                )
                
        # Reset submit flag
        if st.session_state.submit_pressed:
            st.session_state.submit_pressed = False
                
    except Exception as e:
        st.error(str(e))
else:
    st.info("👆 Please load your documents using the sidebar to start chatting!")

# Add a footer
st.markdown("---")
st.markdown(
    "Made with ❤️ using Streamlit, LangChain, and Qdrant. "
    "Upload your documents and start chatting!"
)

def process_documents(documents: List[str]) -> List[str]:
    """Process documents into chunks."""
    chunks = []
    for doc in documents:
        if doc and isinstance(doc, str):  # Verify it's a non-empty string
            # Split into smaller chunks (adjust chunk_size as needed)
            chunk_size = 1000
            for i in range(0, len(doc), chunk_size):
                chunk = doc[i:i + chunk_size].strip()
                if chunk:  # Only add non-empty chunks
                    chunks.append(chunk)
    
    # Debug the chunks
    st.write(f"Debug - Total chunks created: {len(chunks)}")
    if chunks:
        st.write("Debug - First chunk preview:", chunks[0][:100])
    
    if not chunks:
        raise ValueError("No valid text chunks were created from the documents")
        
    return chunks