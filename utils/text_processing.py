import tiktoken
from typing import List, Dict, Any

def chunk_tokens(text: str, model_type: str, api_key: str = None) -> List[str]:
    """
    Chunk text into smaller pieces based on token limit
    
    Args:
        text (str): The text to chunk
        model_type (str): The model type for token counting
        api_key (str): The API key for Anthropic
        
    Returns:
        list: List of text chunks
    """
    # Use tiktoken for all token counting
    encoding = tiktoken.get_encoding("cl100k_base")
    
    if model_type.lower() == "anthropic":
        max_tokens = 100000  # Claude's context window
    else:  # OpenAI
        max_tokens = 4096
        
    token_count = len(encoding.encode(text))

    chunks = []
    tokens = encoding.encode(text, disallowed_special=())

    while tokens:
        chunk = tokens[:max_tokens]
        chunk_text = encoding.decode(chunk)
        last_punctuation = max(
            chunk_text.rfind("."),
            chunk_text.rfind("?"),
            chunk_text.rfind("!"),
            chunk_text.rfind("\n"),
        )
        if last_punctuation != -1:
            chunk_text = chunk_text[: last_punctuation + 1]
        cleaned_text = chunk_text.replace("\n", " ").strip()

        if cleaned_text and (not cleaned_text.isspace()):
            chunks.append(cleaned_text)
        tokens = tokens[len(encoding.encode(chunk_text, disallowed_special=())):]

    return chunks

def process_documents(documents: List[str], chunk_size: int = 100, model_type: str = "openai", api_key: str = None) -> List[str]:
    """
    Process a list of documents into chunks
    
    Args:
        documents (List[str]): List of document texts
        chunk_size (int): Size of each chunk in tokens
        model_type (str): Model type for token counting
        api_key (str): API key for the model
        
    Returns:
        List[str]: Processed chunks
    """
    chunks = []
    for doc in documents:
        if doc and isinstance(doc, str):  # Verify it's a non-empty string
            # Use tiktoken directly instead of model-specific methods
            encoding = tiktoken.get_encoding("cl100k_base")
            
            # Split into smaller chunks (adjust chunk_size as needed)
            chunk_size = 1000
            doc_tokens = encoding.encode(doc)
            
            for i in range(0, len(doc_tokens), chunk_size):
                chunk_tokens = doc_tokens[i:i + chunk_size]
                chunk = encoding.decode(chunk_tokens).strip()
                if chunk:  # Only add non-empty chunks
                    chunks.append(chunk)
    
    # Debug the chunks
    import streamlit as st
    st.write(f"Debug - Total chunks created: {len(chunks)}")
    if chunks:
        st.write("Debug - First chunk preview:", chunks[0][:100])
    
    if not chunks:
        raise ValueError("No valid text chunks were created from the documents")
        
    return chunks