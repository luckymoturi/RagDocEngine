"""
Hybrid Retrieval System for IntelliDoc
Combines embedding-based retrieval with ML model processing and keyword search
"""

import google.generativeai as genai
from typing import List, Dict, Any, Tuple
import asyncio
from datetime import datetime
import os
from dotenv import load_dotenv
import re
from collections import Counter

load_dotenv()


class HybridRetrieval:
    def __init__(self, embedding_service, chroma_handler):
        """Initialize hybrid retrieval system"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.embedding_service = embedding_service
        self.chroma_handler = chroma_handler
    
    async def hybrid_search(
        self, 
        query: str, 
        document_id: str, 
        n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining:
        1. Embedding-based semantic search
        2. Keyword-based BM25 search
        3. ML model relevance scoring
        """
        
        # Step 1: Embedding-based retrieval
        query_embedding = self.embedding_service.generate_query_embedding(query)
        semantic_results = self.chroma_handler.similarity_search(
            query_embedding=query_embedding,
            document_id=document_id,
            n_results=n_results
        )
        
        # Step 2: Keyword-based retrieval
        keywords = self._extract_keywords(query)
        keyword_results = self.chroma_handler.keyword_search(
            keywords=keywords,
            document_id=document_id,
            n_results=n_results
        )
        
        # Step 3: Combine and deduplicate results
        combined_results = self._merge_results(semantic_results, keyword_results)
        
        # Step 4: ML model re-ranking
        reranked_results = await self._ml_rerank(query, combined_results)
        
        return reranked_results[:n_results]
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query"""
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'what', 'which', 'who',
            'when', 'where', 'why', 'how', 'this', 'that', 'these', 'those'
        }
        
        # Tokenize and clean
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords
    
    def _merge_results(
        self, 
        semantic_results: List[Dict], 
        keyword_results: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Merge and deduplicate results from different retrieval methods"""
        
        # Create a dictionary to track unique chunks
        merged = {}
        
        # Add semantic results with their scores
        for result in semantic_results:
            chunk_id = result['metadata'].get('chunk_id', '')
            doc_id = result['metadata'].get('document_id', '')
            key = f"{doc_id}_{chunk_id}"
            
            if key not in merged:
                merged[key] = {
                    'chunk_text': result['chunk_text'],
                    'metadata': result['metadata'],
                    'semantic_score': result.get('similarity_score', 0),
                    'keyword_score': 0,
                    'combined_score': 0
                }
        
        # Add keyword results and update scores
        for result in keyword_results:
            chunk_id = result['metadata'].get('chunk_id', '')
            doc_id = result['metadata'].get('document_id', '')
            key = f"{doc_id}_{chunk_id}"
            
            if key in merged:
                merged[key]['keyword_score'] = result.get('keyword_score', 0)
            else:
                merged[key] = {
                    'chunk_text': result['chunk_text'],
                    'metadata': result['metadata'],
                    'semantic_score': 0,
                    'keyword_score': result.get('keyword_score', 0),
                    'combined_score': 0
                }
        
        # Calculate combined scores (weighted average)
        for key in merged:
            semantic = merged[key]['semantic_score']
            keyword = merged[key]['keyword_score']
            # Weight: 70% semantic, 30% keyword
            merged[key]['combined_score'] = (semantic * 0.7) + (keyword * 0.3)
        
        # Convert to list and sort by combined score
        results = list(merged.values())
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return results
    
    async def _ml_rerank(
        self, 
        query: str, 
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Use ML model to re-rank results based on relevance"""
        
        if not results:
            return results
        
        # Prepare context for ML model
        chunks_text = "\n\n".join([
            f"[Chunk {i+1}]\n{result['chunk_text'][:500]}"
            for i, result in enumerate(results[:10])  # Limit to top 10 for efficiency
        ])
        
        prompt = f"""
        Analyze the relevance of the following document chunks to the user's query.
        Rate each chunk's relevance on a scale of 0-10.
        
        User Query: {query}
        
        Document Chunks:
        {chunks_text}
        
        Provide your ratings in this exact format:
        Chunk 1: [score]
        Chunk 2: [score]
        ...
        
        Only provide the ratings, no explanations.
        """
        
        try:
            response = self.model.generate_content(prompt)
            ml_scores = self._parse_ml_scores(response.text)
            
            # Update results with ML scores
            for i, result in enumerate(results[:len(ml_scores)]):
                ml_score = ml_scores.get(i, 5.0) / 10.0  # Normalize to 0-1
                # Combine with existing score: 50% combined, 50% ML
                result['ml_score'] = ml_score
                result['final_score'] = (result['combined_score'] * 0.5) + (ml_score * 0.5)
            
            # Sort by final score
            results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
            
        except Exception as e:
            print(f"ML re-ranking failed: {str(e)}")
            # Fall back to combined scores
            for result in results:
                result['final_score'] = result['combined_score']
        
        return results
    
    def _parse_ml_scores(self, response_text: str) -> Dict[int, float]:
        """Parse ML model's relevance scores"""
        scores = {}
        
        # Extract scores using regex
        pattern = r'Chunk (\d+):\s*(\d+(?:\.\d+)?)'
        matches = re.findall(pattern, response_text)
        
        for match in matches:
            chunk_num = int(match[0]) - 1  # Convert to 0-indexed
            score = float(match[1])
            scores[chunk_num] = score
        
        return scores
    
    async def compare_retrieval_methods(
        self, 
        query: str, 
        document_id: str
    ) -> Dict[str, Any]:
        """
        Compare different retrieval methods and return analytics
        Useful for debugging and optimization
        """
        
        # Get results from each method
        query_embedding = self.embedding_service.generate_query_embedding(query)
        
        semantic_results = self.chroma_handler.similarity_search(
            query_embedding=query_embedding,
            document_id=document_id,
            n_results=5
        )
        
        keywords = self._extract_keywords(query)
        keyword_results = self.chroma_handler.keyword_search(
            keywords=keywords,
            document_id=document_id,
            n_results=5
        )
        
        hybrid_results = await self.hybrid_search(query, document_id, n_results=5)
        
        return {
            "query": query,
            "extracted_keywords": keywords,
            "semantic_results": {
                "count": len(semantic_results),
                "top_scores": [r.get('similarity_score', 0) for r in semantic_results[:3]],
                "chunks": [r['chunk_text'][:100] + "..." for r in semantic_results[:3]]
            },
            "keyword_results": {
                "count": len(keyword_results),
                "top_scores": [r.get('keyword_score', 0) for r in keyword_results[:3]],
                "chunks": [r['chunk_text'][:100] + "..." for r in keyword_results[:3]]
            },
            "hybrid_results": {
                "count": len(hybrid_results),
                "top_scores": [r.get('final_score', 0) for r in hybrid_results[:3]],
                "chunks": [r['chunk_text'][:100] + "..." for r in hybrid_results[:3]]
            },
            "recommendation": self._get_recommendation(semantic_results, keyword_results, hybrid_results)
        }
    
    def _get_recommendation(
        self, 
        semantic: List[Dict], 
        keyword: List[Dict], 
        hybrid: List[Dict]
    ) -> str:
        """Provide recommendation on which method works best"""
        
        if not semantic and not keyword:
            return "No results found. Try rephrasing your query."
        
        semantic_avg = sum(r.get('similarity_score', 0) for r in semantic) / max(len(semantic), 1)
        keyword_avg = sum(r.get('keyword_score', 0) for r in keyword) / max(len(keyword), 1)
        hybrid_avg = sum(r.get('final_score', 0) for r in hybrid) / max(len(hybrid), 1)
        
        if hybrid_avg > max(semantic_avg, keyword_avg) * 1.1:
            return "Hybrid search provides best results for this query."
        elif semantic_avg > keyword_avg * 1.2:
            return "Semantic search is most effective for this query."
        elif keyword_avg > semantic_avg * 1.2:
            return "Keyword search is most effective for this query."
        else:
            return "All methods provide similar quality results."

