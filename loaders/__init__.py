# contents of loaders/__init__.py here
from .notion_loader import NotionLoader
from .pdf_loader import PDFLoader
from .docx_loader import DocxLoader

__all__ = ['NotionLoader', 'PDFLoader', 'DocxLoader']