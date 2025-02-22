from typing import List, Optional, Tuple, Dict
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatAnthropic
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores.qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.models import Distance, VectorParams
import streamlit as st

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
                timeout=60,  # Increase timeout
                prefer_grpc=False  # Force HTTP
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
                vectors_config=rest.VectorParams(
                    size=1536,
                    distance=rest.Distance.COSINE
                )
            )
            
        return client
    except Exception as e:
        st.error(f"Connection error details: {str(e)}")
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
    """
    try:
        # Use OpenAI embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        
        # Create vectors
        vectors = embeddings.embed_documents(texts)
        
        # Upload directly using client
        client.upsert(
            collection_name=collection_name,
            points=rest.Batch(
                ids=[str(i) for i in range(len(texts))],
                vectors=vectors,
                payloads=[{"text": text} for text in texts]
            )
        )
    except Exception as e:
        st.error(f"Data loading error details: {str(e)}")
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