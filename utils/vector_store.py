from typing import List, Optional, Tuple, Dict
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.models import Distance, VectorParams
import streamlit as st
from uuid import uuid4
from langchain.schema import Document

def connect_to_vectorstore(
    host: str,
    port: Optional[int] = None,
    api_key: Optional[str] = None,
    collection_name: str = "documents_collection",
    openai_api_key: Optional[str] = None
) -> Tuple[QdrantClient, OpenAIEmbeddings]:
    """
    Connect to Qdrant vector store
    """
    try:
        st.write(f"Connecting to Qdrant at: {host}")
        
        # Initialize embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        if host.startswith('http'):
            client = QdrantClient(
                url=host,
                api_key=api_key,
                timeout=60,
                prefer_grpc=False
            )
        else:
            client = QdrantClient(
                host=host,
                port=port
            )
            
        # Recreate collection
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
            
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
            
        return client, embeddings
            
    except Exception as e:
        st.error(str(e))
        raise

def load_data_into_vectorstore(
    client: QdrantClient,
    texts: List[str],
    api_key: str,
    collection_name: str = "documents_collection",
    connection_params: Optional[Dict] = None
) -> None:
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        
        # Filter and prepare texts
        valid_texts = [text for text in texts if text and text.strip()]
        st.write(f"Debug - Processing {len(valid_texts)} valid texts")
        
        # Create Document objects
        documents = [
            Document(
                page_content=text,
                metadata={"source": f"doc_{i}"}
            ) 
            for i, text in enumerate(valid_texts)
        ]
        
        # Create Qdrant wrapper
        qdrant = Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=embeddings
        )
        
        # Add documents using Langchain's method
        qdrant.add_documents(documents)
            
        st.success(f"Successfully loaded {len(valid_texts)} documents")
        
    except Exception as e:
        st.error(str(e))
        raise

def load_chain(client: QdrantClient, api_key: str,
               collection_name: str = "documents_collection",
               model_type: str = "openai", model_name: str = "gpt-3.5-turbo"):
    """
    Load ConversationalRetrievalChain with specified LLM
    
    Args:
        client (QdrantClient): Qdrant client
        api_key (str): API key for the chosen model provider
        collection_name (str): Vector store collection name
        model_type (str): Type of model to use ('openai' or 'anthropic')
        model_name (str): Specific model name
        
    Returns:
        ConversationalRetrievalChain: Chain for question answering
    """
    # Use OpenAI embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = Qdrant(
        client=client, 
        collection_name=collection_name, 
        embeddings=embeddings
    )
    
    if model_type.lower() == "anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model=model_name, anthropic_api_key=api_key, temperature=0.0)
    else:  # default to OpenAI
        llm = ChatOpenAI(temperature=0.0, model_name=model_name, openai_api_key=api_key)
        
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )
    return chain