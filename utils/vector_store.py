from typing import List, Optional
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatAnthropic
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores.qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models

def connect_to_vectorstore(
    host: str,
    port: Optional[int] = None,
    api_key: Optional[str] = None,
    collection_name: str = "documents_collection"
) -> QdrantClient:
    """
    Connect to Qdrant vector store
    
    Args:
        host (str): Qdrant host address
        port (Optional[int]): Port number for local connection
        api_key (Optional[str]): API key for cloud connection
        collection_name (str): Name of the collection
        
    Returns:
        QdrantClient: Configured Qdrant client
    """
    try:
        if host.startswith('http'):
            # Cloud connection
            return QdrantClient(
                url=host,
                api_key=api_key
            )
        else:
            # Local connection
            return QdrantClient(
                host=host,
                port=port
            )
    except Exception as e:
        raise Exception(f"Error connecting to Qdrant: {str(e)}")

def load_data_into_vectorstore(
    client: QdrantClient,
    texts: List[str],
    api_key: str,
    collection_name: str = "documents_collection"
) -> None:
    """
    Load text chunks into Qdrant vector store
    
    Args:
        client (QdrantClient): Qdrant client
        texts (List[str]): List of text chunks to load
        api_key (str): OpenAI API key for embeddings
        collection_name (str): Name of the collection
    """
    try:
        # Use OpenAI embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        
        # Get client configuration
        client_config = client.config
        
        # For cloud Qdrant
        if hasattr(client_config, 'uri'):
            Qdrant.from_texts(
                texts=texts,
                embedding=embeddings,
                url=client_config.uri,
                api_key=client_config.api_key,
                collection_name=collection_name,
                force_recreate=True
            )
        # For local Qdrant
        else:
            Qdrant.from_texts(
                texts=texts,
                embedding=embeddings,
                host=client_config.host,
                port=client_config.port,
                collection_name=collection_name,
                force_recreate=True
            )
    except Exception as e:
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