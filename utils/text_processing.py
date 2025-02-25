import tiktoken
from typing import List, Dict, Any

def chunk_tokens(text: str, model_type: str) -> List[str]:
    """
    Chunk text into smaller pieces based on token limit
    
    Args:
        text (str): The text to chunk
        model_type (str): The model type for token counting
        
    Returns:
        list: List of text chunks
    """
    if model_type.lower() == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        max_tokens = 100000  # Claude's context window
        messages = [{"role": "user", "content": text}]
        token_count = client.messages.count_tokens(messages=messages)
    else:  # OpenAI
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        max_tokens = 4096
        token_count = len(encoding.encode(text))

    chunks = []
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text, disallowed_special=())

    while tokens:
        chunk = tokens[:max_tokens]
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
        chunks.extend(chunk_tokens(doc, "anthropic"))
    return chunks