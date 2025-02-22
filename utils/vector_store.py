from typing import List, Optional, Tuple, Dict
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatAnthropic
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores.qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models
import streamlit as st
from qdrant_client.http.models import VectorParams, Distance

def connect_to_vectorstore(
    host: str,
    port: Optional[int] = None,
    api_key: Optional[str] = None,
    collection_name: str = "documents_collection"
) -> QdrantClient:
    """
    Connect to Qdrant vector store
    """
    try:
        # Debug logging
        st.write(f"Connecting to Qdrant at: {host}")
        
        if host.startswith('http'):
            # Cloud connection
            client = QdrantClient(
                url=host,
                api_key=api_key,
                prefer_grpc=False  # Force HTTP instead of gRPC
            )
        else:
            # Local connection
            client = QdrantClient(
                host=host,
                port=port
            )
            
        # Try to get or create collection
        try:
            client.get_collection(collection_name)
        except Exception:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
            
        return client
    except Exception as e:
        raise Exception(f"Failed to connect to Qdrant: {str(e)}")

def load_data_into_vectorstore(
    client: QdrantClient,
    texts: List[str],
    api_key: str,
    collection_name: str = "documents_collection",
    connection_params: Dict = None
) -> None:
    """
    Load text chunks into Qdrant vector store
    
    Args:
        client (QdrantClient): Qdrant client
        texts (List[str]): List of text chunks to load
        api_key (str): OpenAI API key for embeddings
        collection_name (str): Name of the collection
        connection_params (Dict): Connection parameters
    """
    try:
        # Debug logging
        print(f"Loading data with params: {connection_params}")
        
        # Use OpenAI embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        
        if connection_params['is_cloud']:
            print("Using cloud configuration for loading data")
            Qdrant.from_texts(
                texts=texts,
                embedding=embeddings,
                url=connection_params['host'],
                api_key=connection_params['api_key'],
                collection_name=collection_name,
                force_recreate=True
            )
        else:
            print("Using local configuration for loading data")
            Qdrant.from_texts(
                texts=texts,
                embedding=embeddings,
                host=connection_params['host'],
                port=connection_params['port'],
                collection_name=collection_name,
                force_recreate=True
            )
    except Exception as e:
        print(f"Data loading error: {str(e)}")
        raise Exception(f"Error loading data into vector store: {str(e)}")

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