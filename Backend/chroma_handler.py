import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import json
import re

class ChromaHandler:
    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize ChromaDB client with persistent storage"""
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get the main collection
        self.collection = self.client.get_or_create_collection(
            name="intellidoc_documents",
            metadata={"description": "Document chunks with embeddings"}
        )
    
    def store_document(
        self, 
        document_name: str, 
        chunks: List[Dict[str, Any]], 
        embeddings: List[List[float]], 
        metadata: Dict[str, Any]
    ) -> str:
        """Store document chunks and embeddings in ChromaDB"""
        
        document_id = str(uuid.uuid4())
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        chunk_embeddings = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{document_id}_chunk_{i}"
            ids.append(chunk_id)
            documents.append(chunk['text'])
            
            # Combine chunk metadata with document metadata
            chunk_metadata = {
                "document_id": document_id,
                "document_name": document_name,
                "chunk_id": i,
                "chunk_length": chunk['length'],
                "chunk_type": chunk.get('type', 'text'),
                "upload_time": datetime.now().isoformat(),
                **metadata  # Include document metadata
            }
            
            # ChromaDB metadata values must be strings, ints, floats, or bools
            clean_metadata = self._clean_metadata(chunk_metadata)
            metadatas.append(clean_metadata)
            chunk_embeddings.append(embedding)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=chunk_embeddings,
            metadatas=metadatas
        )
        
        return document_id
    
    def similarity_search(
        self, 
        query_embedding: List[float], 
        document_id: Optional[str] = None, 
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Perform similarity search with optional document filtering and quality scoring"""
        
        where_filter = {}
        if document_id:
            where_filter["document_id"] = document_id
        
        # Get more results than requested to allow for filtering
        search_results = min(n_results * 2, 20)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=search_results,
            where=where_filter if where_filter else None,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format and score results
        formatted_results = []
        if results["documents"]:
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0], 
                results["distances"][0]
            )):
                similarity_score = 1 - distance  # Convert distance to similarity
                
                # Apply quality scoring
                quality_score = self._calculate_chunk_quality(doc, similarity_score)
                
                formatted_results.append({
                    "chunk_text": doc,
                    "metadata": metadata,
                    "similarity_score": similarity_score,
                    "quality_score": quality_score,
                    "distance": distance,
                    "relevance_rank": i + 1
                })
        
        # Sort by quality score (combination of similarity and content quality)
        formatted_results.sort(key=lambda x: x["quality_score"], reverse=True)
        
        # Update relevance ranks after sorting
        for i, result in enumerate(formatted_results[:n_results]):
            result["relevance_rank"] = i + 1
        
        return formatted_results[:n_results]
    
    def _calculate_chunk_quality(self, text: str, similarity_score: float) -> float:
        """Calculate quality score for a chunk based on content and similarity"""
        # Base score from similarity
        quality = similarity_score
        
        # Boost for informative content
        if len(text) > 100:  # Prefer longer, more informative chunks
            quality += 0.1
        
        # Boost for structured content (contains keywords that suggest good content)
        quality_indicators = [
            'definition', 'explanation', 'example', 'process', 'method', 
            'analysis', 'result', 'conclusion', 'important', 'key'
        ]
        
        text_lower = text.lower()
        indicator_count = sum(1 for indicator in quality_indicators if indicator in text_lower)
        quality += indicator_count * 0.05
        
        # Penalize very short or repetitive content
        words = text.split()
        if len(words) < 10:
            quality -= 0.2
        elif len(set(words)) / len(words) < 0.5:  # High repetition
            quality -= 0.1
        
        return max(0.0, min(1.0, quality))  # Clamp between 0 and 1
    
    def keyword_search(
        self, 
        keywords: List[str], 
        document_id: Optional[str] = None,
        n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Advanced keyword search with ranking"""
        
        # Create a combined query from keywords
        query_text = " ".join(keywords)
        
        where_filter = {}
        if document_id:
            where_filter["document_id"] = document_id
        
        # First, get all documents for the specified document_id
        all_results = self.collection.get(
            where=where_filter if where_filter else None,
            include=["documents", "metadatas"]
        )
        
        # Perform keyword matching and scoring
        scored_results = []
        
        for i, (doc, metadata) in enumerate(zip(all_results["documents"], all_results["metadatas"])):
            score = self._calculate_keyword_score(doc, keywords)
            
            if score > 0:  # Only include results with keyword matches
                scored_results.append({
                    "chunk_text": doc,
                    "metadata": metadata,
                    "keyword_score": score,
                    "matched_keywords": self._find_matched_keywords(doc, keywords)
                })
        
        # Sort by score and return top results
        scored_results.sort(key=lambda x: x["keyword_score"], reverse=True)
        return scored_results[:n_results]
    
    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document"""
        results = self.collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"]
        )
        
        chunks = []
        if results["documents"]:
            for doc, metadata in zip(results["documents"], results["metadatas"]):
                chunks.append({
                    "text": doc,
                    "metadata": metadata
                })
        
        return chunks
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all stored documents with metadata"""
        results = self.collection.get(include=["metadatas"])
        
        # Group by document_id and get unique documents
        documents = {}
        
        for metadata in results["metadatas"]:
            doc_id = metadata.get("document_id")
            if doc_id and doc_id not in documents:
                documents[doc_id] = {
                    "document_id": doc_id,
                    "document_name": metadata.get("document_name", "Unknown"),
                    "upload_time": metadata.get("upload_time", ""),
                    "pages": metadata.get("pages", 0),
                    "word_count": metadata.get("word_count", 0),
                    "document_type": metadata.get("document_type", "Unknown"),
                    "stored_file_path": metadata.get("stored_file_path"),
                    "original_filename": metadata.get("original_filename")
                }
        
        return list(documents.values())
    
    def check_document_exists(self, filename: str) -> Optional[Dict[str, Any]]:
        """Check if a document with the given filename already exists"""
        try:
            results = self.collection.get(include=["metadatas"])
            
            # Search for document with matching original_filename
            for metadata in results["metadatas"]:
                if metadata.get("original_filename") == filename:
                    doc_id = metadata.get("document_id")
                    return {
                        "document_id": doc_id,
                        "document_name": metadata.get("document_name", "Unknown"),
                        "upload_time": metadata.get("upload_time", ""),
                        "pages": metadata.get("pages", 0),
                        "word_count": metadata.get("word_count", 0),
                        "summary": metadata.get("summary", ""),
                        "stored_file_path": metadata.get("stored_file_path"),
                        "original_filename": metadata.get("original_filename")
                    }
            
            return None
            
        except Exception as e:
            print(f"Error checking document existence: {str(e)}")
            return None
    
    def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a specific document"""
        try:
            # Get all chunk IDs for this document
            results = self.collection.get(
                where={"document_id": document_id},
                include=["metadatas"]
            )
            
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                return True
            return False
            
        except Exception as e:
            print(f"Error deleting document: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": self.collection.name
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Clean metadata to ensure ChromaDB compatibility"""
        clean_meta = {}
        
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                clean_meta[key] = value
            elif isinstance(value, (list, dict)):
                clean_meta[key] = json.dumps(value)
            else:
                clean_meta[key] = str(value)
        
        return clean_meta
    
    def _calculate_keyword_score(self, text: str, keywords: List[str]) -> float:
        """Calculate relevance score based on keyword matches"""
        text_lower = text.lower()
        score = 0.0
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Exact matches get higher score
            exact_matches = len(re.findall(r'\b' + re.escape(keyword_lower) + r'\b', text_lower))
            score += exact_matches * 2.0
            
            # Partial matches get lower score
            partial_matches = text_lower.count(keyword_lower) - exact_matches
            score += partial_matches * 0.5
        
        # Normalize by text length
        if len(text) > 0:
            score = score / (len(text) / 1000)  # Score per 1000 characters
        
        return score
    
    def _find_matched_keywords(self, text: str, keywords: List[str]) -> List[str]:
        """Find which keywords matched in the text"""
        text_lower = text.lower()
        matched = []
        
        for keyword in keywords:
            if keyword.lower() in text_lower:
                matched.append(keyword)
        
        return matched