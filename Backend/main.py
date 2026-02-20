from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import tempfile
import uvicorn
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

from pdf_extractor import PDFExtractor
from embedding_service import EmbeddingService
from chroma_handler import ChromaHandler
from chat_handler import ChatHandler
from smart_insights import SmartInsights

# Try to import TTS service, make it optional
try:
    from tts_service import TTSService
    TTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: TTS service not available. Install gtts to enable text-to-speech: pip install gtts")
    print(f"Error: {e}")
    TTS_AVAILABLE = False
    TTSService = None

app = FastAPI(title="IntelliDoc API", version="1.0.0")

# CORS middleware
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "https://rag-doc-engine-5jbg.vercel.app,http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Initialize services
pdf_extractor = PDFExtractor()
embedding_service = EmbeddingService()
chroma_handler = ChromaHandler()
chat_handler = ChatHandler(embedding_service, chroma_handler)
smart_insights = SmartInsights()

# Initialize TTS service if available
if TTS_AVAILABLE:
    tts_service = TTSService()
else:
    tts_service = None

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    document_id: Optional[str] = None  # Optional: None = search all documents
    compare_mode: bool = False  # Enable to get both RAG and ML-only responses

class DocumentSummary(BaseModel):
    document_id: str
    summary_type: str = "comprehensive"

class KeywordSearch(BaseModel):
    keywords: List[str]
    document_id: str

class Citation(BaseModel):
    document_id: str
    query: str

class ConversationHistory(BaseModel):
    document_id: str

class ConversationSearch(BaseModel):
    document_id: str
    search_term: str

class TTSRequest(BaseModel):
    text: str
    lang: str = "en"
    slow: bool = False

class TranslateRequest(BaseModel):
    text: str
    target_lang: str

@app.get("/")
async def root():
    """Root endpoint for health check"""
    return {
        "message": "IntelliDoc API is running", 
        "status": "healthy",
        "features": {
            "tts_available": TTS_AVAILABLE
        }
    }

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process PDF document"""
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Check if document already exists
        existing_doc = chroma_handler.check_document_exists(file.filename)
        if existing_doc:
            print(f"INFO: Document '{file.filename}' already exists. Returning existing document_id: {existing_doc['document_id']}")
            
            # Count chunks for this document
            chunks = chroma_handler.get_document_chunks(existing_doc['document_id'])
            
            return {
                "document_id": existing_doc['document_id'],
                "filename": file.filename,
                "pages": existing_doc.get('pages', 0),
                "total_chunks": len(chunks),
                "summary": existing_doc.get('summary', '')[:500] + "...",
                "already_exists": True,
                "message": "Document already processed. Using existing embeddings."
            }
        
        # Create uploads directory if it doesn't exist
        uploads_dir = "uploads"
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Generate unique filename
        import uuid
        unique_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        stored_filename = f"{unique_id}{file_extension}"
        stored_path = os.path.join(uploads_dir, stored_filename)
        
        # Save uploaded file permanently
        content = await file.read()
        with open(stored_path, 'wb') as f:
            f.write(content)
        
        # Also create temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Extract text and metadata
        extraction_result = pdf_extractor.extract_text_and_metadata(tmp_path)
        
        # Generate embeddings
        embeddings = embedding_service.generate_embeddings(extraction_result['chunks'])
        print(embeddings)
        
        # Add file path to metadata
        extraction_result['metadata']['stored_file_path'] = stored_path
        extraction_result['metadata']['original_filename'] = file.filename
        
        print(f"DEBUG: Stored PDF file at: {stored_path}")
        print(f"DEBUG: File exists after storage: {os.path.exists(stored_path)}")
        print(f"DEBUG: Metadata with file path: {extraction_result['metadata']}")
        
        # Store in ChromaDB
        document_id = chroma_handler.store_document(
            document_name=file.filename,
            chunks=extraction_result['chunks'],
            embeddings=embeddings,
            metadata=extraction_result['metadata']
        )
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return {
            "document_id": document_id,
            "filename": file.filename,
            "pages": extraction_result['metadata']['pages'],
            "total_chunks": len(extraction_result['chunks']),
            "summary": extraction_result['metadata']['summary'][:500] + "..."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_with_document(chat_request: ChatMessage):
    """
    Chat with document using RAG
    Set compare_mode=true to get both RAG and ML-only responses for comparison
    """
    try:
        response = await chat_handler.process_query(
            query=chat_request.message,
            document_id=chat_request.document_id,
            compare_mode=chat_request.compare_mode
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
async def summarize_document(summary_request: DocumentSummary):
    """Generate different types of summaries"""
    try:
        summary = await chat_handler.generate_summary(
            document_id=summary_request.document_id,
            summary_type=summary_request.summary_type
        )
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/keyword-search")
async def keyword_search(search_request: KeywordSearch):
    """Advanced keyword search with semantic similarity"""
    try:
        results = chroma_handler.keyword_search(
            keywords=search_request.keywords,
            document_id=search_request.document_id
        )
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-citations")
async def generate_citations(citation_request: Citation):
    """Generate academic citations for referenced content"""
    try:
        citations = await chat_handler.generate_citations(
            query=citation_request.query,
            document_id=citation_request.document_id
        )
        return {"citations": citations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.options("/documents")
async def documents_options():
    """Handle preflight requests for documents endpoint"""
    return {"message": "OK"}

@app.get("/documents")
async def list_documents():
    """List all uploaded documents"""
    try:
        documents = chroma_handler.list_documents()
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/document/{document_id}/analytics")
async def document_analytics(document_id: str):
    """Get document analytics and insights"""
    try:
        analytics = await chat_handler.get_document_analytics(document_id)
        return analytics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/document/{document_id}/smart-insights")
async def get_smart_insights(document_id: str):
    """
    Get AI-powered smart insights for a document
    Includes: key insights, questions, connections, learning path
    """
    try:
        # Get document chunks
        chunks = chroma_handler.get_document_chunks(document_id)
        if not chunks:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get document metadata
        documents = chroma_handler.list_documents()
        document = next((doc for doc in documents if doc["document_id"] == document_id), None)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Extract text from chunks
        chunk_texts = [chunk['text'] for chunk in chunks]
        
        # Generate insights
        insights = await smart_insights.generate_document_insights(
            chunk_texts,
            document
        )
        
        # Generate smart questions
        questions = await smart_insights.generate_smart_questions(chunk_texts)
        
        # Generate quick facts
        facts = await smart_insights.generate_quick_facts(chunk_texts)
        
        # Get other documents for connections
        other_docs = [
            {"name": doc["document_name"], "summary": doc.get("summary", "")[:500]}
            for doc in documents
            if doc["document_id"] != document_id
        ]
        
        # Find connections
        connections = await smart_insights.generate_document_connections(
            chunk_texts,
            other_docs
        )
        
        # Generate learning path
        learning_path = await smart_insights.generate_learning_path(chunk_texts)
        
        return {
            "document_id": document_id,
            "document_name": document["document_name"],
            "insights": insights,
            "smart_questions": questions,
            "quick_facts": facts,
            "connections": connections,
            "learning_path": learning_path,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/document/{document_id}/pdf-available")
async def check_pdf_available(document_id: str):
    """Check if PDF viewing is available for this document"""
    try:
        documents = chroma_handler.list_documents()
        document = next((doc for doc in documents if doc["document_id"] == document_id), None)
        
        if not document:
            return {"available": False, "reason": "Document not found"}
        
        stored_file_path = document.get("stored_file_path")
        if not stored_file_path:
            return {"available": False, "reason": "Document uploaded before PDF viewing was implemented"}
        
        if not os.path.exists(stored_file_path):
            return {"available": False, "reason": "PDF file not found on server"}
        
        return {"available": True, "reason": "PDF viewing available"}
        
    except Exception as e:
        return {"available": False, "reason": str(e)}

@app.get("/document/{document_id}/pdf")
async def get_document_pdf(document_id: str):
    """Serve PDF file for viewing"""
    try:
        # Get document metadata to find the stored file
        documents = chroma_handler.list_documents()
        document = next((doc for doc in documents if doc["document_id"] == document_id), None)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get the stored file path from metadata
        print(f"DEBUG: Document metadata: {document}")
        stored_file_path = document.get("stored_file_path")
        print(f"DEBUG: Looking for stored_file_path: {stored_file_path}")
        
        if not stored_file_path:
            raise HTTPException(
                status_code=404, 
                detail=f"PDF file not available for viewing. This document was uploaded before PDF viewing was implemented. Please re-upload the document to enable PDF viewing."
            )
        
        if not os.path.exists(stored_file_path):
            print(f"DEBUG: File does not exist at path: {stored_file_path}")
            # List files in uploads directory for debugging
            uploads_dir = "uploads"
            if os.path.exists(uploads_dir):
                files = os.listdir(uploads_dir)
                print(f"DEBUG: Files in uploads directory: {files}")
            raise HTTPException(status_code=404, detail=f"PDF file not found at path: {stored_file_path}")
        
        # Serve the PDF file for inline viewing
        
        # Read the file content
        with open(stored_file_path, 'rb') as f:
            pdf_content = f.read()
        
        # Return response with inline disposition
        return Response(
            content=pdf_content,
            media_type='application/pdf',
            headers={
                'Content-Disposition': 'inline',  # This makes it display inline instead of download
                'Content-Type': 'application/pdf'
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/document/{document_id}/download")
async def download_document_pdf(document_id: str):
    """Download PDF file"""
    try:
        # Get document metadata to find the stored file
        documents = chroma_handler.list_documents()
        document = next((doc for doc in documents if doc["document_id"] == document_id), None)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        stored_file_path = document.get("stored_file_path")
        if not stored_file_path or not os.path.exists(stored_file_path):
            raise HTTPException(status_code=404, detail="PDF file not found on server")
        
        from fastapi.responses import FileResponse
        return FileResponse(
            path=stored_file_path,
            media_type='application/pdf',
            filename=document.get("original_filename", "document.pdf")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
                

@app.get("/document/{document_id}/conversations")
async def get_conversation_history(document_id: str):
    """Get conversation history for a document"""
    try:
        conversations = chat_handler.get_conversation_history(document_id)
        return {"conversations": conversations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/document/{document_id}/conversations")
async def clear_conversation_history(document_id: str):
    """Clear conversation history for a document"""
    try:
        success = chat_handler.clear_conversation_history(document_id)
        if success:
            return {"message": "Conversation history cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear conversation history")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/document/{document_id}/conversations/{conversation_id}")
async def delete_conversation(document_id: str, conversation_id: str):
    """Delete a specific conversation"""
    try:
        success = chat_handler.delete_conversation(document_id, conversation_id)
        if success:
            return {"message": "Conversation deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Conversation not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/document/{document_id}/conversations/search")
async def search_conversations(document_id: str, q: str):
    """Search conversations by content"""
    try:
        results = chat_handler.search_conversations(document_id, q)
        return {"results": results, "total": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/document/{document_id}/conversations/stats")
async def get_conversation_stats(document_id: str):
    """Get conversation statistics for a document"""
    try:
        stats = chat_handler.get_conversation_stats(document_id)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/document/{document_id}/conversations/export")
async def export_conversations(document_id: str, format: str = "json"):
    """Export conversations in different formats"""
    try:
        if format not in ["json", "text"]:
            raise HTTPException(status_code=400, detail="Format must be 'json' or 'text'")
        
        exported_data = chat_handler.export_conversations(document_id, format)
        
        if format == "json":
            return JSONResponse(content={"data": exported_data})
        else:  # text format
            from fastapi.responses import PlainTextResponse
            return PlainTextResponse(content=exported_data, media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/document/{document_id}/conversations/charts")
async def get_conversations_with_charts(document_id: str):
    """Get all conversations that include chart data"""
    try:
        conversations = chat_handler.conversation_storage.get_conversations_with_charts(document_id)
        chart_types = chat_handler.conversation_storage.get_chart_types_used(document_id)
        return {
            "conversations": conversations,
            "chart_types_used": chart_types,
            "total_charts": len(conversations)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/document/{document_id}")
async def delete_document(document_id: str):
    """Delete document and its embeddings"""
    try:
        chroma_handler.delete_document(document_id)
        chat_handler.clear_conversation_history(document_id)  # Also clear conversations
        return {"message": "Document deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/text-to-speech")
async def text_to_speech(tts_request: TTSRequest):
    """Convert text to speech and return audio file"""
    if not TTS_AVAILABLE or tts_service is None:
        raise HTTPException(
            status_code=503, 
            detail="Text-to-speech service is not available. Please install gtts: pip install gtts"
        )
    
    try:
        audio_path = tts_service.text_to_speech(
            text=tts_request.text,
            lang=tts_request.lang,
            slow=tts_request.slow
        )
        
        # Read the audio file
        with open(audio_path, 'rb') as audio_file:
            audio_content = audio_file.read()
        
        # Return audio file
        from fastapi.responses import Response
        return Response(
            content=audio_content,
            media_type='audio/mpeg',
            headers={
                'Content-Disposition': 'inline; filename="speech.mp3"',
                'Content-Type': 'audio/mpeg'
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate")
async def translate_text(translate_request: TranslateRequest):
    """Translate text to target language"""
    try:
        translated_text = await chat_handler.translate_text(
            text=translate_request.text,
            target_language=translate_request.target_lang
        )
        return {"translated_text": translated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tts/cache-stats")
async def get_tts_cache_stats():
    """Get TTS cache statistics"""
    if not TTS_AVAILABLE or tts_service is None:
        raise HTTPException(status_code=503, detail="TTS service not available")
    
    try:
        stats = tts_service.get_cache_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts/clean-cache")
async def clean_tts_cache(max_files: int = 100):
    """Clean TTS cache"""
    if not TTS_AVAILABLE or tts_service is None:
        raise HTTPException(status_code=503, detail="TTS service not available")
    
    try:
        tts_service.clean_cache(max_files=max_files)
        return {"message": "Cache cleaned successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Create necessary directories on startup
uploads_dir = "uploads"
conversations_dir = "conversations"
tts_cache_dir = "tts_cache"
os.makedirs(uploads_dir, exist_ok=True)
os.makedirs(conversations_dir, exist_ok=True)
os.makedirs(tts_cache_dir, exist_ok=True)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)