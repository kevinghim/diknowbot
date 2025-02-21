from typing import List, Optional
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatAnthropic
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores.qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models

def connect_to_vectorstore(host: str = "localhost", port: int = 6333, 
                           collection_name: str = "documents_collection"):
    """
    Connect to Qdrant vector store
    
    Args:
        host (str): Qdrant host
        port (int): Qdrant port
        collection_name (str): Name of the collection
        
    Returns:
        QdrantClient: Initialized Qdrant client
    """
    client = QdrantClient(host=host, port=port)
    try:
        client.get_collection(collection_name)
    except Exception as e:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
        )
    return client

def load_data_into_vectorstore(client: QdrantClient, docs: List[str], 
                              openai_api_key: str, 
                              collection_name: str = "documents_collection"):
    """
    Load documents into vector store
    
    Args:
        client (QdrantClient): Qdrant client
        docs (List[str]): List of document chunks
        openai_api_key (str): OpenAI API key
        collection_name (str): Collection name
        
    Returns:
        list: List of document IDs
    """
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    qdrant_client = Qdrant(
        client=client, 
        collection_name=collection_name, 
        embeddings=embeddings
    )
    ids = qdrant_client.add_texts(docs)
    return ids

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