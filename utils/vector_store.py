from typing import List, Optional, Tuple, Dict
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.models import Distance, VectorParams
import streamlit as st
from uuid import uuid4
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory

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

def load_data_into_vectorstore(client, chunks, api_key, collection_name, qdrant_config, model_type="openai"):
    """
    Load chunks into vector store
    
    Args:
        client: Qdrant client
        chunks: Text chunks to load
        api_key: API key for embeddings
        collection_name: Name of the collection
        qdrant_config: Configuration for Qdrant
        model_type: Type of model (openai or anthropic)
    """
    try:
        # Initialize embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        
        # Create Qdrant wrapper
        vectorstore = Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=embeddings
        )
        
        # Add documents to vector store
        vectorstore.add_texts(chunks)
        
        return vectorstore
    except Exception as e:
        raise Exception(f"Failed to load data into vector store: {str(e)}")

def load_chain(client, collection_name, embeddings, model_type="openai", model_name="gpt-3.5-turbo", api_key=None):
    """Load a conversational retrieval chain."""
    # Create Qdrant wrapper
    vectorstore = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings
    )
    
    # Create the chat model
    if model_type.lower() == "anthropic":
        from langchain_anthropic import ChatAnthropic
        try:
            llm = ChatAnthropic(
                model=model_name,
                anthropic_api_key=api_key,
                temperature=0.7
            )
        except Exception as e:
            st.error(f"Error initializing Anthropic: {str(e)}")
            # Fallback to OpenAI
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7
            )
    else:  # default to OpenAI
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            temperature=0.7
        )
    
    # Create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create the conversation chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )
    
    return qa_chain