import tiktoken
from typing import List, Dict, Any

def chunk_tokens(text: str, token_limit: int) -> list:
    """
    Chunk text into smaller pieces based on token limit
    
    Args:
        text (str): The text to chunk
        token_limit (int): Maximum tokens per chunk
        
    Returns:
        list: List of text chunks
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")

    chunks = []
    tokens = tokenizer.encode(text, disallowed_special=())

    while tokens:
        chunk = tokens[:token_limit]
        chunk_text = tokenizer.decode(chunk)
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
        tokens = tokens[len(tokenizer.encode(chunk_text, disallowed_special=())):]

    return chunks

def process_documents(documents: List[str], chunk_size: int = 100) -> List[str]:
    """
    Process a list of documents into chunks
    
    Args:
        documents (List[str]): List of document texts
        chunk_size (int): Size of each chunk in tokens
        
    Returns:
        List[str]: Processed chunks
    """
    chunks = []
    for doc in documents:
        chunks.extend(chunk_tokens(doc, chunk_size))
    return chunks