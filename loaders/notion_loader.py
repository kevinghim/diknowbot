import requests
from typing import List, Dict, Any
from utils.document_valuation import evaluate_document_value

class NotionLoader:
    """Loader for Notion documents"""
    
    BASE_URL = "https://api.notion.com"
    
    def __init__(self, notion_api_key: str):
        """
        Initialize Notion loader
        
        Args:
            notion_api_key (str): Notion API key
        """
        self.headers = {
            "Authorization": f"Bearer {notion_api_key}", 
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28"
        }
    
    def get_blocks(self, page_id: str) -> Dict[str, Any]:
        """
        Get blocks from a Notion page
        
        Args:
            page_id (str): Notion page ID
            
        Returns:
            Dict[str, Any]: JSON response with page blocks
        """
        res = requests.get(
            f"{self.BASE_URL}/v1/blocks/{page_id}/children?page_size=100", 
            headers=self.headers
        )
        return res.json()
    
    def search(self, query: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Search Notion
        
        Args:
            query (Dict[str, Any], optional): Search query. Defaults to {}.
            
        Returns:
            Dict[str, Any]: Search results
        """
        if query is None:
            query = {}
        res = requests.post(
            f"{self.BASE_URL}/v1/search", 
            headers=self.headers, 
            json=query
        )
        return res.json()
    
    def get_page_text(self, page_id: str) -> List[str]:
        """
        Get text content from a Notion page
        
        Args:
            page_id (str): Notion page ID
            
        Returns:
            List[str]: Page text content
        """
        page_text = []
        blocks = self.get_blocks(page_id)
        for item in blocks['results']:
            item_type = item.get('type')
            content = item.get(item_type)
            if content and content.get('rich_text'):
                for text in content.get('rich_text'):
                    plain_text = text.get('plain_text')
                    page_text.append(plain_text)
        return page_text
    
    def load_documents(self) -> List[str]:
        """
        Load all documents from Notion
        
        Returns:
            List[str]: List of document contents
        """
        documents = []
        all_notion_documents = self.search()
        items = all_notion_documents.get('results', [])
        
        for item in items:
            object_type = item.get('object')
            object_id = item.get('id')
            url = item.get('url')
            title = ""
            page_text = []

            if object_type == 'page':
                # Get title
                title_content = item.get('properties', {}).get('title')
                if title_content:
                    title_items = title_content.get('title', [])
                    if title_items and len(title_items) > 0:
                        title = title_items[0].get('text', {}).get('content', '')
                elif item.get('properties', {}).get('Name'):
                    name_items = item.get('properties', {}).get('Name', {}).get('title', [])
                    if len(name_items) > 0:
                        title = name_items[0].get('text', {}).get('content', '')

                # Get content
                page_text.append(title)
                page_content = self.get_page_text(object_id)
                page_text.extend(page_content)

                # Flatten list and join with periods
                flat_list = [item for sublist in page_text for item in sublist]
                text_per_page = ". ".join(flat_list)
                
                if len(text_per_page) > 0:
                    documents.append(text_per_page)
                    # Evaluate document value
                    if title:
                        doc_name = f"Notion: {title}"
                    else:
                        doc_name = f"Notion page {object_id}"
                    
                    value_info = evaluate_document_value(text_per_page, doc_name)
                    if 'document_values' in st.session_state:
                        st.session_state['document_values'][doc_name] = value_info

        return documents