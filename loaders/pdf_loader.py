import os
import PyPDF2
from typing import List, Dict, Optional
import pytesseract
from PIL import Image
import io
import tempfile

class PDFLoader:
    """Loader for PDF documents"""
    def __init__(self, ocr_enabled: bool = False):
        self.ocr_enabled = ocr_enabled
        """
        Initialize PDF loader
        
        Args:
            ocr_enabled (bool): Whether to use OCR for scanned PDFs
        """
        
    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text from PDF file
        
        Args:
            file_path (str): Path to PDF file
            
        Returns:
            str: Extracted text
        """
        text = ""
        try:
            # Try standard PDF text extraction
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    
                    # If page is empty and OCR is enabled, try OCR
                    if not page_text.strip() and self.ocr_enabled:
                        page_text = self._ocr_pdf_page(file_path, page_num)
                        
                    text += page_text + "\n\n"
                    
            return text
        except Exception as e:
            print(f"Error extracting text from PDF {file_path}: {str(e)}")
            return ""
            
    def _ocr_pdf_page(self, file_path: str, page_num: int) -> str:
        """
        Perform OCR on a PDF page
        
        Args:
            file_path (str): Path to PDF file
            page_num (int): Page number
            
        Returns:
            str: Extracted text via OCR
        """
        try:
            # Convert PDF page to image
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_img:
                temp_img_path = temp_img.name
                
            # Use PyMuPDF or another library to convert PDF to image
            # For simplicity, we'll assume this step works
            # (In a real implementation, you'd use a library like PyMuPDF)
            
            # Apply OCR to the image
            text = pytesseract.image_to_string(Image.open(temp_img_path))
            
            # Clean up
            os.unlink(temp_img_path)
            
            return text
        except Exception as e:
            print(f"OCR error on {file_path} page {page_num}: {str(e)}")
            return ""
    
    def load_documents(self, directory_path: str) -> List[Dict[str, str]]:
        """
        Load all PDF documents from a directory
        
        Args:
            directory_path (str): Path to directory containing PDFs
            
        Returns:
            List[Dict[str, str]]: List of documents with metadata
        """
        documents = []
        
        # Check if directory exists
        if not os.path.isdir(directory_path):
            print(f"Directory {directory_path} does not exist")
            return documents
            
        # Process all PDF files in the directory
        for filename in os.listdir(directory_path):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(directory_path, filename)
                try:
                    text = self.extract_text_from_pdf(file_path)
                    if text:
                        documents.append({
                            'content': text,
                            'source': file_path,
                            'filename': filename
                        })
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    
        return documents
        
    def load_document(self, file_path: str) -> Optional[Dict[str, str]]:
        """
        Load a single PDF document
        
        Args:
            file_path (str): Path to PDF file
            
        Returns:
            Optional[Dict[str, str]]: Document with metadata if successful
        """
        if not os.path.isfile(file_path) or not file_path.lower().endswith('.pdf'):
            print(f"Invalid PDF file: {file_path}")
            return None
            
        try:
            text = self.extract_text_from_pdf(file_path)
            if text:
                return {
                    'content': text,
                    'source': file_path,
                    'filename': os.path.basename(file_path)
                }
            return None
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None