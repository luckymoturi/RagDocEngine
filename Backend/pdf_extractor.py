import PyPDF2
import fitz  # PyMuPDF
import re
from typing import List, Dict, Any
import google.generativeai as genai
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class PDFExtractor:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def extract_text_and_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text, metadata, and structure from PDF"""
        try:
            # Open PDF with PyMuPDF for better text extraction
            doc = fitz.open(pdf_path)

            full_text = ""
            pages_text = []
            images_info = []
            tables_info = []

            for page_num in range(len(doc)):
                page = doc[page_num]

                # Extract text
                page_text = page.get_text()
                pages_text.append({
                    "page": page_num + 1,
                    "text": page_text,
                    "word_count": len(page_text.split())
                })
                full_text += page_text + "\n"

                # Extract images
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    images_info.append({
                        "page": page_num + 1,
                        "image_index": img_index,
                        "size": (img[2], img[3]) if len(img) > 3 else None
                    })

                # Extract tables (basic detection)
                tables = self._detect_tables(page_text)
                if tables:
                    tables_info.extend([{
                        "page": page_num + 1,
                        "table_index": i,
                        "content": table
                    } for i, table in enumerate(tables)])

            # Extract metadata
            metadata = self._extract_metadata(doc, full_text, len(doc), images_info, tables_info)

            # Chunk the text intelligently
            chunks = self._intelligent_chunking(full_text, pages_text)

            doc.close()
            
            return {
                "text": full_text,
                "pages": pages_text,
                "chunks": chunks,
                "metadata": metadata,
                "images": images_info,
                "tables": tables_info
            }
            
        except Exception as e:
            raise Exception(f"Error extracting PDF: {str(e)}")
    
    def _extract_metadata(self, doc, full_text: str, page_count: int, images: List, tables: List) -> Dict[str, Any]:
        """Extract comprehensive metadata from PDF"""
        # Basic metadata from PDF
        pdf_metadata = doc.metadata
        
        # Text analysis
        word_count = len(full_text.split())
        char_count = len(full_text)
        
        # Generate AI summary
        summary = self._generate_summary(full_text[:4000])  # Limit for API
        
        # Extract key topics using regex patterns
        topics = self._extract_topics(full_text)
        
        # Detect document type
        doc_type = self._detect_document_type(full_text)
        
        # Reading time estimation (average 200 words per minute)
        reading_time = max(1, word_count // 200)
        
        return {
            "title": pdf_metadata.get("title", "Unknown"),
            "author": pdf_metadata.get("author", "Unknown"),
            "subject": pdf_metadata.get("subject", ""),
            "creator": pdf_metadata.get("creator", ""),
            "creation_date": pdf_metadata.get("creationDate", ""),
            "pages": page_count,
            "word_count": word_count,
            "char_count": char_count,
            "images_count": len(images),
            "tables_count": len(tables),
            "estimated_reading_time": f"{reading_time} minutes",
            "summary": summary,
            "topics": topics,
            "document_type": doc_type,
            "processing_date": datetime.now().isoformat()
        }
    
    def _generate_summary(self, text: str) -> str:
        """Generate AI summary using Gemini"""
        try:
            prompt = f"""
            Provide a comprehensive summary of the following document text in 3-4 sentences.
            Focus on the main topics, key findings, and purpose of the document.
            
            Text: {text}
            """
            response = self.model.generate_content(prompt)
            return response.text
        except Exception:
            return "Summary generation failed. Please try again."
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract key topics using keyword analysis"""
        # Simple topic extraction using common patterns
        topics = []
        
        # Look for section headers (capitalized words at line start)
        headers = re.findall(r'^[A-Z][A-Z\s]{2,}$', text, re.MULTILINE)
        topics.extend([h.strip() for h in headers[:10]])
        
        # Look for bullet points or numbered items that might be topics
        bullets = re.findall(r'(?:^|\n)[\d\-\*\•]\s+([A-Z][^.\n]{10,50})', text)
        topics.extend(bullets[:10])
        
        return list(set(topics))  # Remove duplicates
    
    def _detect_document_type(self, text: str) -> str:
        """Detect the type of document based on content"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['research', 'methodology', 'results', 'conclusion', 'abstract']):
            return "Research Paper"
        elif any(word in text_lower for word in ['report', 'executive summary', 'findings']):
            return "Report"
        elif any(word in text_lower for word in ['manual', 'instructions', 'guide', 'how to']):
            return "Manual/Guide"
        elif any(word in text_lower for word in ['contract', 'agreement', 'terms', 'conditions']):
            return "Legal Document"
        elif any(word in text_lower for word in ['chapter', 'contents', 'index']):
            return "Book/Textbook"
        else:
            return "General Document"
    
    def _detect_tables(self, text: str) -> List[str]:
        """Basic table detection in text"""
        tables = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            # Look for lines with multiple spaces or tabs (potential table rows)
            if re.search(r'\s{3,}|\t{2,}', line) and len(line.split()) > 2:
                # Get surrounding lines for context
                start = max(0, i-2)
                end = min(len(lines), i+3)
                table_text = '\n'.join(lines[start:end])
                if table_text not in tables:
                    tables.append(table_text)
        
        return tables[:5]  # Limit to first 5 tables
    
    def _intelligent_chunking(self, full_text: str, pages_text: List) -> List[Dict[str, Any]]:
        """Create intelligent chunks based on content structure"""
        chunks = []
        chunk_size = 1000  # Target chunk size in characters
        overlap = 200  # Overlap between chunks
        
        # Split by paragraphs first
        paragraphs = [p.strip() for p in full_text.split('\n\n') if p.strip()]
        
        current_chunk = ""
        chunk_id = 0
        
        for para in paragraphs:
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(para) > chunk_size and current_chunk:
                chunks.append({
                    "id": chunk_id,
                    "text": current_chunk.strip(),
                    "length": len(current_chunk.strip()),
                    "type": "paragraph_based"
                })
                
                # Start new chunk with overlap
                current_chunk = current_chunk[-overlap:] + "\n\n" + para
                chunk_id += 1
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append({
                "id": chunk_id,
                "text": current_chunk.strip(),
                "length": len(current_chunk.strip()),
                "type": "paragraph_based"
            })
        
        return chunks