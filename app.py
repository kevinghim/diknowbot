import streamlit as st
from streamlit_chat import message
import os
import tempfile
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI, ChatAnthropic
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

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

from utils.document_valuation import evaluate_document_value

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

if 'document_values' not in st.session_state:
    st.session_state['document_values'] = {}

if 'total_document_value' not in st.session_state:
    st.session_state['total_document_value'] = 0.0
    
if 'document_topics' not in st.session_state:
    st.session_state['document_topics'] = {}

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
col1, col2 = st.columns([1, 14])  # Adjust ratio as needed
with col1:
    st.image("assets/dino_icon.png", width=70)
with col2:
    st.title('Dino the Knowledge Bot')
st.subheader('Ask questions about your Notion, PDF, and Word documents')

# Add this right after your title, before the tabs
col1, col2 = st.columns([1, 3])
with col1:
    st.metric(
        label="Total Document Value", 
        value=f"${st.session_state['total_document_value']:,.2f}",
        delta=None,
        delta_color="normal"
    )
    
    if st.session_state['document_topics']:
        with st.expander("Document Topics"):
            topics_count = {}
            for doc, topics in st.session_state['document_topics'].items():
                for topic in topics:
                    if topic in topics_count:
                        topics_count[topic] += 1
                    else:
                        topics_count[topic] = 1
            
            # Sort topics by count
            sorted_topics = dict(sorted(topics_count.items(), key=lambda x: x[1], reverse=True))
            
            for topic, count in sorted_topics.items():
                st.write(f"{topic} ({count})")

# Set up sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # API Keys
    api_key_tab, file_upload_tab, doc_values_tab, model_config_tab = st.tabs([
        "API Keys", 
        "Upload Files", 
        "Document Values",
        "Model Config"
    ])
    
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
    
    with doc_values_tab:
        st.subheader("Document Value Estimates")
        
        # Add a summary at the top
        st.info(f"Total document value: ${st.session_state['total_document_value']:,.2f} across {len(st.session_state['document_values'])} documents")
        
        # Add topic filter
        all_topics = set()
        for topics in st.session_state['document_topics'].values():
            all_topics.update(topics)
        
        selected_topic = st.selectbox("Filter by topic", ["All Topics"] + sorted(list(all_topics)))
        
        # Display documents
        if st.session_state['document_values']:
            # Sort documents by value for better display
            sorted_docs = dict(sorted(
                st.session_state['document_values'].items(),
                key=lambda x: x[1]['estimated_value'],
                reverse=True
            ))
            
            for filename, value_info in sorted_docs.items():
                # Skip if filtering by topic and document doesn't have that topic
                if selected_topic != "All Topics" and filename in st.session_state['document_topics'] and selected_topic not in st.session_state['document_topics'][filename]:
                    continue
                    
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"📄 {filename}")
                with col2:
                    st.write(f"${value_info['estimated_value']:,.2f}")
                with col3:
                    if filename in st.session_state['document_topics']:
                        st.write(", ".join(st.session_state['document_topics'][filename]))
                
                with st.expander("View value factors"):
                    for factor in value_info['factors']:
                        st.write(f"• {factor}")
                st.divider()
        else:
            st.info("No documents have been evaluated yet.")
            
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
                options=["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307", "claude-3-haiku"],
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

def extract_topics_with_llm(content, doc_name, api_key):
    """Extract main topics from document content using LLM"""
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    
    # Truncate content if too long
    max_chars = 10000
    if len(content) > max_chars:
        content = content[:max_chars] + "..."
    
    # Create prompt
    prompt = ChatPromptTemplate.from_template(
        """You are a document classification expert.
        
        Analyze the following document and identify the top 3-5 main topics it covers.
        Return only a comma-separated list of topics (no numbering, no explanation).
        Be specific but concise with topic names (1-3 words per topic).
        
        Document name: {doc_name}
        
        Document content:
        {content}
        
        Topics:"""
    )
    
    # Create LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        openai_api_key=api_key,
        temperature=0.0
    )
    
    # Get topics
    chain = prompt | llm
    result = chain.invoke({"doc_name": doc_name, "content": content})
    topics = [topic.strip() for topic in result.content.split(",")]
    
    return topics

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
                                    content = pdf_doc['content']
                                    all_documents.append(content)
                                    # Evaluate document value
                                    value_info = evaluate_document_value(content, pdf_file.name)
                                    st.session_state['document_values'][pdf_file.name] = value_info
                                    st.session_state['total_document_value'] += value_info['estimated_value']
                                    topics = extract_topics_with_llm(content, pdf_file.name, openai_api_key)
                                    st.session_state['document_topics'][pdf_file.name] = topics
                                    st.sidebar.success(f"✅ Loaded PDF: {pdf_file.name}")
                                    st.sidebar.info(f"📊 Estimated value: ${value_info['estimated_value']}")
                                    with st.sidebar.expander(f"Value factors for {pdf_file.name}"):
                                        for factor in value_info['factors']:
                                            st.write(f"• {factor}")
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
                                    content = docx_doc['content']
                                    all_documents.append(content)
                                    # Evaluate document value
                                    value_info = evaluate_document_value(content, docx_file.name)
                                    st.session_state['document_values'][docx_file.name] = value_info
                                    st.session_state['total_document_value'] += value_info['estimated_value']
                                    topics = extract_topics_with_llm(content, docx_file.name, openai_api_key)
                                    st.session_state['document_topics'][docx_file.name] = topics
                                    st.sidebar.success(f"✅ Loaded Word document: {docx_file.name}")
                                    st.sidebar.info(f"📊 Estimated value: ${value_info['estimated_value']}")
                                    with st.sidebar.expander(f"Value factors for {docx_file.name}"):
                                        for factor in value_info['factors']:
                                            st.write(f"• {factor}")
                        except Exception as e:
                            st.sidebar.error(f"Error processing Word document {docx_file.name}: {str(e)}")
                
                if all_documents:
                    try:
                        # Process documents into chunks
                        if model_provider == "Anthropic":
                            api_key = anthropic_api_key
                            model_type = "anthropic"
                            # For embeddings, we still need to use OpenAI
                            embedding_api_key = openai_api_key
                        else:
                            api_key = openai_api_key
                            model_type = "openai"
                            embedding_api_key = openai_api_key
                            
                        chunks = process_documents(all_documents, model_type=model_type, api_key=api_key)
                        st.write(f"Debug - Created {len(chunks)} text chunks")
                        
                        # Load chunks into vector store
                        load_data_into_vectorstore(
                            vector_store,
                            chunks,
                            embedding_api_key,  # Use OpenAI API key for embeddings
                            collection_name,
                            {
                                'is_cloud': is_deployed,
                                'host': qdrant_host,
                                'port': qdrant_port,
                                'api_key': qdrant_api_key
                            },
                            model_type=model_type
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

def process_documents(documents: List[str], model_type: str, api_key: str) -> List[str]:
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