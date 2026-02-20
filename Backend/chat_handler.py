import google.generativeai as genai
from typing import List, Dict, Any, Optional
import asyncio
import json
from datetime import datetime
import re
import os
from dotenv import load_dotenv
from conversation_storage import ConversationStorage
from hybrid_retrieval import HybridRetrieval

# Load environment variables
load_dotenv()

# Try to import BERT QA service (optional dependency)
try:
    from bert_qa_service import BertQAService, AnswerComparator
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("BERT QA service not available. Install transformers and sentence-transformers for BERT support.")


class ChatHandler:
    def __init__(self, embedding_service, chroma_handler):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.embedding_service = embedding_service
        self.chroma_handler = chroma_handler
        
        # Initialize hybrid retrieval system
        self.hybrid_retrieval = HybridRetrieval(embedding_service, chroma_handler)
        
        # Initialize persistent conversation storage
        self.conversation_storage = ConversationStorage()
        
        # Conversation memory for context (in-memory cache)
        self.conversation_history = {}
        
        # Toggle for hybrid retrieval (can be enabled/disabled)
        self.use_hybrid_retrieval = os.getenv("USE_HYBRID_RETRIEVAL", "false").lower() == "true"
        
        # Initialize BERT QA service (for ML-based answer comparison)
        self.bert_qa = None
        self.answer_comparator = None
        self.use_bert_comparison = os.getenv("USE_BERT_COMPARISON", "true").lower() == "true"
        
        if BERT_AVAILABLE and self.use_bert_comparison:
            try:
                print("Initializing BERT QA service...")
                self.bert_qa = BertQAService()
                self.answer_comparator = AnswerComparator(self.bert_qa.sentence_model)
                print("BERT QA service initialized successfully!")
            except Exception as e:
                print(f"Failed to initialize BERT QA: {str(e)}")
                self.bert_qa = None
    
    async def process_query(self, query: str, document_id: str = None, compare_mode: bool = False) -> Dict[str, Any]:
        """
        Process user query using RAG (Retrieval Augmented Generation)
        If document_id is None, searches across ALL documents
        Uses BERT QA for comparison and selects the best answer
        """
        try:
            # Choose retrieval method based on configuration
            if self.use_hybrid_retrieval and document_id:
                # Use hybrid retrieval (semantic + keyword + ML re-ranking)
                relevant_chunks = await self.hybrid_retrieval.hybrid_search(
                    query=query,
                    document_id=document_id,
                    n_results=5
                )
            else:
                # Use standard semantic search
                query_embedding = self.embedding_service.generate_query_embedding(query)
                relevant_chunks = self.chroma_handler.similarity_search(
                    query_embedding=query_embedding,
                    document_id=document_id,  # None = search all documents
                    n_results=10 if document_id is None else 5  # More results for multi-doc search
                )
            
            # Classify query type for better response generation
            query_type = self._classify_query(query)
            
            # Check if user is asking for visualization
            chart_data = None
            is_viz_request = self._is_visualization_request(query)
            
            if is_viz_request:
                chart_data = await self._generate_chart_data(query, relevant_chunks, document_id)
            
            # Check if user is asking for audio response
            is_audio_request = self._is_audio_request(query)
            
            # Generate RAG response (with context using Gemini)
            rag_response = await self._generate_contextual_response(
                query=query,
                relevant_chunks=relevant_chunks,
                query_type=query_type,
                document_id=document_id,
                include_chart=chart_data is not None
            )
            
            # Calculate confidence for RAG
            rag_confidence = self._calculate_confidence(relevant_chunks)
            
            # Generate BERT answer and compare with RAG
            bert_answer = None
            comparison_result = None
            selected_source = "rag"
            
            # Debug: Check if BERT is available
            print(f"DEBUG: BERT QA available: {self.bert_qa is not None}")
            print(f"DEBUG: Answer comparator available: {self.answer_comparator is not None}")
            print(f"DEBUG: Relevant chunks: {len(relevant_chunks) if relevant_chunks else 0}")
            
            if self.bert_qa and self.answer_comparator and relevant_chunks:
                try:
                    print("DEBUG: Starting BERT QA...")
                    # Get BERT extractive answer
                    bert_answer = await self.bert_qa.answer_question(
                        question=query,
                        context_chunks=relevant_chunks,
                        top_k=5
                    )
                    print(f"DEBUG: BERT answer received: {bert_answer is not None}")
                    
                    # Print both answers for debugging
                    print("\n" + "="*60)
                    print("RAG vs BERT Comparison")
                    print("="*60)
                    print(f"Question: {query}")
                    print("-"*60)
                    print(f"RAG Answer: {rag_response[:300]}..." if len(rag_response) > 300 else f"RAG Answer: {rag_response}")
                    print(f"RAG Confidence: {rag_confidence:.2f}")
                    print("-"*60)
                    bert_text = bert_answer.get("answer", "N/A") if bert_answer else "N/A"
                    bert_conf = bert_answer.get("confidence", 0) if bert_answer else 0
                    print(f"BERT Answer: {bert_text[:300]}..." if len(bert_text) > 300 else f"BERT Answer: {bert_text}")
                    print(f"BERT Confidence: {bert_conf:.3f}")
                    print("-"*60)
                    
                    # Compare RAG and BERT answers
                    comparison_result = await self.answer_comparator.compare_and_select(
                        question=query,
                        rag_answer={"answer": rag_response, "confidence": rag_confidence},
                        bert_answer=bert_answer,
                        context_chunks=relevant_chunks
                    )
                    
                    # Use the best answer
                    selected_source = comparison_result.get("selected", "rag")
                    
                    # Print comparison result
                    print(f"Selected: {selected_source.upper()}")
                    print(f"Reason: {comparison_result.get('reason', 'N/A')}")
                    if 'scores' in comparison_result:
                        print(f"Scores - RAG: {comparison_result['scores'].get('rag', 0):.3f}, BERT: {comparison_result['scores'].get('bert', 0):.3f}")
                    print("="*60 + "\n")
                    
                except Exception as e:
                    print(f"BERT comparison error: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    # Fall back to RAG on error
                    selected_source = "rag"
            else:
                print("DEBUG: BERT comparison skipped - requirements not met")
            
            # Determine final response based on comparison
            if comparison_result and selected_source == "bert":
                response = comparison_result.get("best_answer", rag_response)
            else:
                response = rag_response
            
            # Generate ML-only response (without context) if compare mode requested
            ml_only_response = None
            if compare_mode:
                ml_only_response = await self._generate_ml_only_response(query)
            
            # Store conversation context (in-memory) - only if document_id is provided
            if document_id:
                self._update_conversation_history(document_id, query, response)
            
            # Save conversation persistently - only if document_id is provided
            conversation_id = None
            if document_id:
                conversation_id = self.conversation_storage.save_conversation(
                    document_id=document_id,
                    query=query,
                    response=response,
                    sources=relevant_chunks,
                    query_type=query_type,
                    confidence=rag_confidence,
                    chart_data=chart_data
                )
            
            # Get unique document names from sources for multi-doc mode
            document_names = []
            if document_id is None and relevant_chunks:
                doc_ids = set()
                for chunk in relevant_chunks:
                    doc_name = chunk.get('metadata', {}).get('document_name', 'Unknown')
                    doc_id = chunk.get('metadata', {}).get('document_id', '')
                    if doc_id and doc_id not in doc_ids:
                        doc_ids.add(doc_id)
                        document_names.append(doc_name)
            
            result = {
                "response": response,
                "rag_response": rag_response,  # RAG answer with context
                "bert_answer": bert_answer.get("answer") if bert_answer else None,  # BERT extractive answer
                "ml_only_response": ml_only_response,  # ML answer without context
                "selected_source": selected_source,  # Which model's answer was selected
                "comparison_result": comparison_result,  # Full comparison details
                "sources": relevant_chunks,
                "query_type": query_type,
                "confidence": rag_confidence,
                "bert_confidence": bert_answer.get("confidence") if bert_answer else None,
                "timestamp": datetime.now().isoformat(),
                "conversation_id": conversation_id,
                "has_audio": is_audio_request,  # Flag to indicate audio should be generated
                "compare_mode": compare_mode,
                "multi_document_mode": document_id is None,
                "documents_used": document_names if document_id is None else []
            }
            
            if chart_data:
                result["chart_data"] = chart_data
            
            return result
            
        except Exception as e:
            return {
                "response": f"I encountered an error processing your query: {str(e)}",
                "error": True
            }
    
    async def translate_text(self, text: str, target_language: str) -> str:
        """Translate text to the target language using Gemini"""
        prompt = f"""
        Translate the following text to {target_language}. 
        Keep the formatting, bullet points, and structure intact. 
        Only return the translated text, nothing else.
        
        Text to translate:
        {text}
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return f"Error during translation: {str(e)}"

    async def generate_summary(self, document_id: str, summary_type: str = "comprehensive") -> str:
        """Generate different types of summaries"""
        try:
            # Get all document chunks
            chunks = self.chroma_handler.get_document_chunks(document_id)
            
            if not chunks:
                return "No document found with the specified ID."
            
            # Combine chunks (limit to avoid token limits)
            combined_text = ""
            for chunk in chunks[:10]:  # Limit to first 10 chunks
                combined_text += chunk["text"] + "\n\n"
            
            # Generate summary based on type
            if summary_type == "executive":
                prompt = self._get_executive_summary_prompt(combined_text)
            elif summary_type == "key_points":
                prompt = self._get_key_points_prompt(combined_text)
            elif summary_type == "abstract":
                prompt = self._get_abstract_prompt(combined_text)
            else:  # comprehensive
                prompt = self._get_comprehensive_summary_prompt(combined_text)
            
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as api_error:
                error_str = str(api_error).lower()
                if '429' in error_str or 'quota' in error_str or 'rate' in error_str:
                    # Return a basic extractive summary from the text
                    sentences = combined_text.split('.')[:10]
                    return f"[Extractive Summary - API quota exceeded]\n\n" + '. '.join(sentences) + "."
                raise
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    async def generate_citations(self, query: str, document_id: str) -> List[Dict[str, Any]]:
        """Generate academic citations for referenced content"""
        try:
            # Get relevant chunks for the query
            query_embedding = self.embedding_service.generate_query_embedding(query)
            relevant_chunks = self.chroma_handler.similarity_search(
                query_embedding=query_embedding,
                document_id=document_id,
                n_results=3
            )
            
            citations = []
            for i, chunk in enumerate(relevant_chunks):
                metadata = chunk["metadata"]
                
                # Create citation in APA format
                citation = {
                    "id": i + 1,
                    "text": chunk["chunk_text"][:200] + "...",
                    "apa_format": self._format_apa_citation(metadata),
                    "page_reference": f"p. {self._estimate_page_number(metadata)}",
                    "relevance_score": chunk["similarity_score"]
                }
                citations.append(citation)
            
            return citations
            
        except Exception as e:
            return [{"error": f"Citation generation failed: {str(e)}"}]
    
    async def get_document_analytics(self, document_id: str) -> Dict[str, Any]:
        """Get comprehensive document analytics"""
        try:
            chunks = self.chroma_handler.get_document_chunks(document_id)
            
            if not chunks:
                return {"error": "Document not found"}
            
            # Extract metadata from first chunk
            doc_metadata = chunks[0]["metadata"]
            
            # Analyze document content
            all_text = " ".join([chunk["text"] for chunk in chunks])
            
            analytics = {
                "basic_info": {
                    "document_name": doc_metadata.get("document_name", "Unknown"),
                    "total_chunks": len(chunks),
                    "total_words": len(all_text.split()),
                    "total_characters": len(all_text),
                    "pages": doc_metadata.get("pages", 0)
                },
                "content_analysis": await self._analyze_content(all_text[:5000]),
                "readability": self._calculate_readability(all_text),
                "key_statistics": self._extract_statistics(all_text),
                "topics": await self._extract_advanced_topics(all_text[:3000]),
                "sentiment": await self._analyze_sentiment(all_text[:2000])
            }
            
            return analytics
            
        except Exception as e:
            return {"error": f"Analytics generation failed: {str(e)}"}
    
    def _is_visualization_request(self, query: str) -> bool:
        """Check if the user is requesting a visualization"""
        visualization_keywords = [
            'visualize', 'chart', 'graph', 'plot', 'show me', 'display',
            'bar chart', 'line chart', 'pie chart', 'histogram', 'diagram',
            'visual representation', 'graphical', 'create a chart', 'make a graph',
            'draw', 'illustrate', 'represent visually', 'show graphically',
            'doughnut chart', 'donut chart', 'statistics chart', 'data visualization',
            'visual analysis', 'graphic', 'infographic', 'dashboard'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in visualization_keywords)
    
    def _is_audio_request(self, query: str) -> bool:
        """Check if the user is requesting audio/voice response"""
        audio_keywords = [
            'read it', 'read this', 'read that', 'read aloud', 'read out',
            'voice', 'audio', 'speak', 'say it', 'tell me', 'listen',
            'hear', 'sound', 'speech', 'narrate', 'vocalize',
            'in voice', 'with voice', 'as audio', 'as speech',
            'play audio', 'audio clip', 'voice clip', 'sound clip',
            'voice response', 'audio response', 'speak it'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in audio_keywords)
    
    async def _generate_chart_data(self, query: str, relevant_chunks: List[Dict], document_id: str) -> Dict[str, Any]:
        """Generate chart data based on the query and document content"""
        try:
            # Combine relevant text for analysis
            context = "\n\n".join([chunk["chunk_text"] for chunk in relevant_chunks[:3]])
            


            
            prompt = f"""
            Based on the user's request: "{query}" and the following document content, generate chart data in JSON format.
            
            Document Content:
            {context}
            
            INSTRUCTIONS:
            1. Analyze the document for numerical data, statistics, categories, or themes
            2. Choose the most appropriate chart type:
               - "bar": For comparing categories or discrete values
               - "line": For trends, time series, or continuous data
               - "pie": For parts of a whole or percentages
               - "doughnut": For modern circular representation of proportions
            3. Create meaningful labels and data points
            4. Use actual numbers from the document when available
            5. If no numbers exist, create representative data based on content analysis
            
            Return ONLY a valid JSON object with this exact structure:
            {{
                "type": "bar",
                "title": "Descriptive Chart Title",
                "labels": ["Category 1", "Category 2", "Category 3"],
                "datasets": [{{
                    "label": "Dataset Description",
                    "data": [25, 45, 30]
                }}]
            }}
            
            Make the chart informative and relevant to the user's request.
            """
            
            response = self.model.generate_content(prompt)
            
            # Try to parse the JSON response
            try:
                import json
                # Clean the response to extract JSON
                response_text = response.text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:-3]
                elif response_text.startswith('```'):
                    response_text = response_text[3:-3]
                
                # Remove any extra text before or after JSON
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    response_text = response_text[json_start:json_end]
                
                chart_data = json.loads(response_text)
                
                # Validate required fields
                required_fields = ['type', 'title', 'labels', 'datasets']
                if all(field in chart_data for field in required_fields):
                    # Add chart URL for visualization
                    chart_data = self._add_chart_url(chart_data)
                    return chart_data
                else:
                    return self._create_default_chart(query, context)
                    
            except json.JSONDecodeError:
                return self._create_default_chart(query, context)
                
        except Exception as e:
            print(f"Error generating chart data: {str(e)}")
            return self._create_default_chart(query, context)
    
    def _create_default_chart(self, query: str, context: str) -> Dict[str, Any]:
        """Create a default chart when AI generation fails"""
        # Extract some basic statistics from the context
        words = context.split()
        word_count = len(words)
        sentences = context.split('.')
        sentence_count = len([s for s in sentences if s.strip()])
        paragraphs = context.split('\n\n')
        paragraph_count = len([p for p in paragraphs if p.strip()])
        
        # Determine chart type based on query
        query_lower = query.lower()
        chart_type = "bar"  # default
        
        if any(word in query_lower for word in ['pie', 'circular', 'proportion']):
            chart_type = "pie"
        elif any(word in query_lower for word in ['line', 'trend', 'over time']):
            chart_type = "line"
        elif any(word in query_lower for word in ['doughnut', 'donut']):
            chart_type = "doughnut"
        
        # Create more meaningful data for visualization
        data_points = [
            min(word_count // 10, 100),  # Scale down words for better visualization
            sentence_count,
            paragraph_count,
            len([w for w in words if len(w) > 6])  # Complex words
        ]
        
        labels = ["Words (x10)", "Sentences", "Paragraphs", "Complex Words"]
        
        chart_data = {
            "type": chart_type,
            "title": "Document Content Analysis",
            "labels": labels,
            "datasets": [{
                "label": "Content Statistics",
                "data": data_points
            }]
        }
        
        # Generate QuickChart URL
        import json
        import urllib.parse
        
        config = {
            "type": chart_type,
            "data": {
                "labels": labels,
                "datasets": [{
                    "label": "Content Statistics",
                    "data": data_points,
                    "backgroundColor": [
                        "rgba(99, 102, 241, 0.8)",
                        "rgba(139, 92, 246, 0.8)",
                        "rgba(6, 182, 212, 0.8)",
                        "rgba(16, 185, 129, 0.8)"
                    ],
                    "borderColor": [
                        "rgb(99, 102, 241)",
                        "rgb(139, 92, 246)",
                        "rgb(6, 182, 212)",
                        "rgb(16, 185, 129)"
                    ],
                    "borderWidth": 2
                }]
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": "Document Content Analysis",
                        "font": {"size": 16, "weight": "bold"}
                    }
                }
            }
        }
        
        encoded_config = urllib.parse.quote(json.dumps(config))
        chart_url = f"https://quickchart.io/chart?c={encoded_config}&width=600&height=400&format=png"
        
        chart_data["chart_url"] = chart_url
        return chart_data
    
    def _add_chart_url(self, chart_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add QuickChart URL to chart data for visualization"""
        try:
            import json
            import urllib.parse
            
            # Create QuickChart configuration
            config = {
                "type": chart_data["type"],
                "data": {
                    "labels": chart_data["labels"],
                    "datasets": chart_data["datasets"]
                },
                "options": {
                    "responsive": True,
                    "plugins": {
                        "title": {
                            "display": True,
                            "text": chart_data["title"],
                            "font": {"size": 16, "weight": "bold"}
                        },
                        "legend": {
                            "display": True,
                            "position": "top"
                        }
                    }
                }
            }
            
            # Add colors if not present
            if chart_data["datasets"]:
                dataset = config["data"]["datasets"][0]
                if "backgroundColor" not in dataset:
                    colors = [
                        "rgba(99, 102, 241, 0.8)",
                        "rgba(139, 92, 246, 0.8)", 
                        "rgba(6, 182, 212, 0.8)",
                        "rgba(16, 185, 129, 0.8)",
                        "rgba(245, 158, 11, 0.8)",
                        "rgba(239, 68, 68, 0.8)",
                        "rgba(168, 85, 247, 0.8)",
                        "rgba(34, 197, 94, 0.8)"
                    ]
                    dataset["backgroundColor"] = colors[:len(chart_data["labels"])]
                    
                if "borderColor" not in dataset:
                    border_colors = [
                        "rgb(99, 102, 241)",
                        "rgb(139, 92, 246)",
                        "rgb(6, 182, 212)", 
                        "rgb(16, 185, 129)",
                        "rgb(245, 158, 11)",
                        "rgb(239, 68, 68)",
                        "rgb(168, 85, 247)",
                        "rgb(34, 197, 94)"
                    ]
                    dataset["borderColor"] = border_colors[:len(chart_data["labels"])]
                    dataset["borderWidth"] = 2
            
            # Generate QuickChart URL
            encoded_config = urllib.parse.quote(json.dumps(config))
            chart_url = f"https://quickchart.io/chart?c={encoded_config}&width=600&height=400&format=png"
            
            # Add URL to chart data
            chart_data["chart_url"] = chart_url
            
            return chart_data
            
        except Exception as e:
            print(f"Error generating chart URL: {str(e)}")
            return chart_data

    def _classify_query(self, query: str) -> str:
        """Classify the type of query for better response generation"""
        query_lower = query.lower()
        
        # Summary queries
        if any(word in query_lower for word in ['summarize', 'summary', 'overview', 'brief', 'outline', 'gist']):
            return "summary"
        
        # Definition queries
        elif any(phrase in query_lower for phrase in ['what is', 'what are', 'define', 'definition', 'meaning of', 'explain what']):
            return "definition"
        
        # Process/How-to queries
        elif any(word in query_lower for word in ['how to', 'how do', 'how can', 'process', 'step', 'method', 'procedure', 'approach']):
            return "process"
        
        # Explanation queries (why/reasoning)
        elif any(word in query_lower for word in ['why', 'reason', 'cause', 'because', 'explain why', 'rationale']):
            return "explanation"
        
        # List queries
        elif any(word in query_lower for word in ['list', 'enumerate', 'bullet', 'points', 'items', 'examples of', 'types of']):
            return "list"
        
        # Comparison queries
        elif any(word in query_lower for word in ['compare', 'comparison', 'difference', 'differences', 'versus', 'vs', 'contrast', 'similar', 'different']):
            return "comparison"
        
        # Analysis queries
        elif any(word in query_lower for word in ['analyze', 'analysis', 'evaluate', 'assessment', 'review', 'examine']):
            return "analysis"
        
        # Specific fact queries
        elif any(word in query_lower for word in ['when', 'where', 'who', 'which', 'how much', 'how many']):
            return "factual"
        
        # Question format
        elif '?' in query:
            return "question"
        
        else:
            return "general"
    
    async def _generate_contextual_response(
        self, 
        query: str, 
        relevant_chunks: List[Dict], 
        query_type: str,
        document_id: str,
        include_chart: bool = False
    ) -> str:
        """Generate contextual response based on query type and retrieved chunks"""
        
        # Get conversation history for context (from persistent storage)
        # Only if document_id is provided (not for multi-document chat)
        conversation_context = []
        if document_id:
            conversation_context = self.conversation_storage.get_recent_conversations(document_id, limit=3)
        
        # Prepare context from relevant chunks
        context = "\n\n".join([chunk["chunk_text"] for chunk in relevant_chunks[:3]])
        
        # Enhanced prompts with follow-up options and detailed responses
        base_instructions = f"""
        You are an intelligent document assistant. Your goal is to provide helpful, detailed, and actionable responses.
        
        RESPONSE GUIDELINES:
        1. Provide comprehensive answers with relevant examples when possible
        2. Structure your response clearly with headings or bullet points when appropriate
        3. Include specific details and context from the document
        4. Adapt your explanation to be both accessible and thorough
        5. Always end with 2-3 helpful follow-up options
        {"6. A chart visualization has been generated to accompany your response" if include_chart else ""}
        
        FOLLOW-UP OPTIONS FORMAT:
        End your response with:
        
        **Would you like me to:**
        • [Specific follow-up option 1 related to the topic]
        • [Specific follow-up option 2 for deeper understanding]
        • {"Create a visual representation of this data" if not include_chart else "Show different chart types for this data"}
        """
        
        # Create enhanced prompts based on query type
        if query_type == "summary":
            prompt = f"""
            {base_instructions}
            
            USER QUERY: "{query}"
            
            DOCUMENT CONTENT:
            {context}
            
            TASK: Provide a comprehensive summary that addresses the user's query. Include:
            - Main themes and key points
            - Important details and context
            - Relevant examples or data from the document
            - Clear structure with headings or bullet points
            
            Make your summary both thorough and easy to understand.
            """
        elif query_type == "definition":
            prompt = f"""
            {base_instructions}
            
            USER QUERY: "{query}"
            
            DOCUMENT CONTENT:
            {context}
            
            TASK: Provide a detailed explanation and definition. Include:
            - Clear, comprehensive definition
            - Context and background information
            - Specific examples from the document
            - How it relates to other concepts mentioned
            - Practical implications or applications
            
            Explain in a way that's both thorough and accessible.
            """
        elif query_type == "process" or query_type == "explanation":
            prompt = f"""
            {base_instructions}
            
            USER QUERY: "{query}"
            
            DOCUMENT CONTENT:
            {context}
            
            TASK: Provide a detailed step-by-step explanation. Include:
            - Clear breakdown of the process or reasoning
            - Specific steps or stages involved
            - Examples and context from the document
            - Why each step is important
            - Potential challenges or considerations
            
            Structure your response logically and include relevant details.
            """
        elif query_type == "list":
            prompt = f"""
            {base_instructions}
            
            USER QUERY: "{query}"
            
            DOCUMENT CONTENT:
            {context}
            
            TASK: Create a comprehensive, well-organized list. Include:
            - Clear, numbered or bulleted format
            - Detailed descriptions for each item
            - Relevant examples or context
            - Additional insights or explanations
            - Logical organization and grouping
            
            Make each list item informative and actionable.
            """
        elif query_type == "comparison":
            prompt = f"""
            {base_instructions}
            
            USER QUERY: "{query}"
            
            DOCUMENT CONTENT:
            {context}
            
            TASK: Provide a detailed comparison. Include:
            - Clear side-by-side analysis
            - Key similarities and differences
            - Specific examples from the document
            - Advantages and disadvantages
            - Context and implications
            
            Structure your comparison clearly and provide thorough analysis.
            """
        elif query_type == "analysis":
            prompt = f"""
            {base_instructions}
            
            USER QUERY: "{query}"
            
            DOCUMENT CONTENT:
            {context}
            
            TASK: Provide a comprehensive analysis. Include:
            - Detailed examination of the topic
            - Key insights and observations
            - Supporting evidence from the document
            - Implications and significance
            - Critical evaluation of important points
            
            Structure your analysis logically with clear reasoning.
            """
        elif query_type == "factual":
            prompt = f"""
            {base_instructions}
            
            USER QUERY: "{query}"
            
            DOCUMENT CONTENT:
            {context}
            
            TASK: Provide specific, factual information. Include:
            - Direct answer to the factual question
            - Supporting context and details
            - Relevant data or statistics from the document
            - Additional related information
            - Source references within the document
            
            Be precise and comprehensive in your factual response.
            """
        else:
            prompt = f"""
            {base_instructions}
            
            USER QUERY: "{query}"
            
            DOCUMENT CONTENT:
            {context}
            
            PREVIOUS CONVERSATION:
            {self._format_conversation_context(conversation_context)}
            
            TASK: Provide a comprehensive, detailed answer. Include:
            - Direct response to the user's question
            - Supporting details and context from the document
            - Relevant examples or evidence
            - Additional insights or implications
            - Clear explanations of complex concepts
            
            If information is not available in the document, clearly state this and suggest what additional information might be helpful.
            
            Adapt your response to be both thorough and accessible, providing value beyond just answering the immediate question.
            """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            error_str = str(e).lower()
            # Check if it's a quota/rate limit error
            if '429' in error_str or 'quota' in error_str or 'rate' in error_str or 'resource_exhausted' in error_str:
                print(f"Gemini API quota exceeded, using BERT fallback...")
                # Fall back to BERT if available
                if self.bert_qa and relevant_chunks:
                    try:
                        bert_result = await self.bert_qa.answer_question(
                            question=query,
                            context_chunks=relevant_chunks,
                            top_k=3
                        )
                        if bert_result and not bert_result.get("error"):
                            return f"[BERT Response] {bert_result.get('answer', 'Unable to generate answer.')}"
                    except Exception as bert_error:
                        print(f"BERT fallback also failed: {str(bert_error)}")
                
                # If BERT also fails, return a helpful message with context
                if relevant_chunks:
                    context_summary = "\n".join([chunk["chunk_text"][:300] for chunk in relevant_chunks[:2]])
                    return f"API quota exceeded. Here's relevant content from your document:\n\n{context_summary}\n\n(Please try again later for AI-generated analysis)"
                return "API quota exceeded. Please try again later or check your API billing."
            
            print(f"Error generating response: {str(e)}")
            return f"I encountered an error generating the response: {str(e)}"
    
    async def _generate_ml_only_response(self, query: str) -> str:
        """
        Generate response using ML model only (without document context)
        This is for comparison purposes to show the difference between RAG and pure ML
        """
        prompt = f"""
        You are an AI assistant. Answer the following question based on your general knowledge.
        Do NOT make up specific details about documents you haven't seen.
        If you don't have specific information, acknowledge that.
        
        Question: {query}
        
        Provide a helpful response based on your general knowledge, but be clear about what you know vs. what would require specific document context.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            error_str = str(e).lower()
            if '429' in error_str or 'quota' in error_str or 'rate' in error_str:
                return "API quota exceeded. ML-only response unavailable."
            print(f"Error generating ML-only response: {str(e)}")
            return f"Error generating ML-only response: {str(e)}"
    
    def _calculate_confidence(self, relevant_chunks: List[Dict]) -> float:
        """Calculate confidence score based on similarity scores and content quality"""
        if not relevant_chunks:
            return 0.0
        
        # Weight by both similarity score and chunk quality
        total_score = 0
        total_weight = 0
        
        for chunk in relevant_chunks:
            similarity = chunk["similarity_score"]
            chunk_length = len(chunk["chunk_text"])
            
            # Higher weight for longer, more similar chunks
            weight = similarity * (1 + min(chunk_length / 1000, 1))
            total_score += similarity * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        confidence = (total_score / total_weight) * 100
        
        # Apply thresholds for better interpretation
        if confidence >= 0.7 * 100:
            return min(confidence, 95.0)  # High confidence
        elif confidence >= 0.4 * 100:
            return confidence * 0.8  # Medium confidence
        else:
            return max(confidence * 0.5, 5.0)  # Low confidence
    
    def _update_conversation_history(self, document_id: str, query: str, response: str):
        """Update conversation history for context"""
        if document_id not in self.conversation_history:
            self.conversation_history[document_id] = []
        
        self.conversation_history[document_id].append({
            "query": query,
            "response": response[:200] + "...",  # Store truncated response
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 5 interactions to manage memory
        if len(self.conversation_history[document_id]) > 5:
            self.conversation_history[document_id] = self.conversation_history[document_id][-5:]
    
    def _format_conversation_context(self, context: List[Dict]) -> str:
        """Format conversation history for prompt context"""
        if not context:
            return "No previous conversation."
        
        formatted = "Recent conversation:\n"
        for item in context[-2:]:  # Last 2 interactions
            # Handle both old format (query/response) and new format (full conversation objects)
            query = item.get('query', '')
            response = item.get('response', '')
            
            # Truncate long responses for context
            if len(response) > 200:
                response = response[:200] + "..."
            
            formatted += f"Q: {query}\nA: {response}\n\n"
        
        return formatted
    
    def _get_executive_summary_prompt(self, text: str) -> str:
        return f"""
        Create an executive summary of the following document. Focus on:
        1. Main objectives and purpose
        2. Key findings or conclusions
        3. Important recommendations or outcomes
        4. Business impact or significance
        
        Document text:
        {text}
        
        Keep the summary concise but comprehensive, suitable for executive-level review.
        """
    
    def _get_key_points_prompt(self, text: str) -> str:
        return f"""
        Extract and list the key points from the following document:
        
        {text}
        
        Present as a bulleted list of the most important information, insights, or conclusions.
        """
    
    def _get_abstract_prompt(self, text: str) -> str:
        return f"""
        Create an academic abstract for the following document content:
        
        {text}
        
        Include: Purpose, methodology (if applicable), key findings, and conclusions.
        Keep it concise and academic in tone.
        """
    
    def _get_comprehensive_summary_prompt(self, text: str) -> str:
        return f"""
        Provide a comprehensive summary of the following document:
        
        {text}
        
        Include all major topics, key information, and important details while maintaining readability.
        """
    
    def _format_apa_citation(self, metadata: Dict[str, Any]) -> str:
        """Format citation in APA style"""
        author = metadata.get("author", "Unknown Author")
        title = metadata.get("document_name", "Unknown Title")
        date = metadata.get("upload_time", datetime.now().isoformat())[:4]  # Extract year
        
        return f"{author} ({date}). {title}."
    
    def _estimate_page_number(self, metadata: Dict[str, Any]) -> int:
        """Estimate page number from chunk metadata"""
        chunk_id = metadata.get("chunk_id", 0)
        # Rough estimation: assume ~3 chunks per page
        return max(1, (chunk_id // 3) + 1)
    
    async def _analyze_content(self, text: str) -> Dict[str, Any]:
        """Analyze document content using AI"""
        try:
            prompt = f"""
            Analyze the following document content and provide:
            1. Main themes (3-5)
            2. Document complexity level (Basic/Intermediate/Advanced)
            3. Target audience
            4. Content type (Technical, Academic, Business, etc.)
            
            Text: {text}
            
            Respond in JSON format.
            """
            
            response = self.model.generate_content(prompt)
            # Try to parse JSON, fallback to text analysis
            try:
                return json.loads(response.text)
            except:
                return {"analysis": response.text}
                
        except Exception:
            return {"error": "Content analysis failed"}
    
    def _calculate_readability(self, text: str) -> Dict[str, Any]:
        """Calculate basic readability metrics"""
        words = text.split()
        sentences = text.split('.')
        
        avg_words_per_sentence = len(words) / max(len(sentences), 1)
        avg_chars_per_word = sum(len(word) for word in words) / max(len(words), 1)
        
        # Simple readability score (simplified Flesch formula)
        readability_score = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * (avg_chars_per_word / 4.7))
        
        if readability_score >= 90:
            level = "Very Easy"
        elif readability_score >= 80:
            level = "Easy"
        elif readability_score >= 70:
            level = "Fairly Easy"
        elif readability_score >= 60:
            level = "Standard"
        elif readability_score >= 50:
            level = "Fairly Difficult"
        elif readability_score >= 30:
            level = "Difficult"
        else:
            level = "Very Difficult"
        
        return {
            "score": round(readability_score, 2),
            "level": level,
            "avg_words_per_sentence": round(avg_words_per_sentence, 2),
            "avg_chars_per_word": round(avg_chars_per_word, 2)
        }
    
    def _extract_statistics(self, text: str) -> Dict[str, Any]:
        """Extract numerical statistics from text"""
        # Find numbers and percentages
        numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', text)
        percentages = re.findall(r'\b\d+(?:\.\d+)?%\b', text)
        
        return {
            "numbers_found": len(numbers),
            "percentages_found": len(percentages),
            "sample_numbers": numbers[:5],
            "sample_percentages": percentages[:3]
        }
    
    async def _extract_advanced_topics(self, text: str) -> List[str]:
        """Extract topics using AI analysis"""
        try:
            prompt = f"""
            Extract the main topics and themes from this text. List 5-8 key topics:
            
            {text}
            
            Respond with just a comma-separated list of topics.
            """
            
            response = self.model.generate_content(prompt)
            topics = [topic.strip() for topic in response.text.split(',')]
            return topics[:8]
            
        except Exception:
            return ["Topic extraction failed"]
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze document sentiment"""
        try:
            prompt = f"""
            Analyze the overall sentiment and tone of this document:
            
            {text}
            
            Provide:
            1. Overall sentiment (Positive/Neutral/Negative)
            2. Tone (Formal/Informal/Academic/Technical/etc.)
            3. Confidence level (1-10)
            
            Respond in JSON format.
            """
            
            response = self.model.generate_content(prompt)
            try:
                return json.loads(response.text)
            except:
                return {"sentiment": "Neutral", "tone": "Unknown", "confidence": 5}
                
        except Exception:
            return {"error": "Sentiment analysis failed"}
    
    def get_conversation_history(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all conversation history for a document"""
        return self.conversation_storage.load_conversations(document_id)
    
    def clear_conversation_history(self, document_id: str) -> bool:
        """Clear all conversation history for a document"""
        # Clear from persistent storage
        success = self.conversation_storage.clear_conversations(document_id)
        
        # Clear from in-memory cache
        if document_id in self.conversation_history:
            del self.conversation_history[document_id]
        
        return success
    
    def delete_conversation(self, document_id: str, conversation_id: str) -> bool:
        """Delete a specific conversation"""
        return self.conversation_storage.delete_conversation(document_id, conversation_id)
    
    def search_conversations(self, document_id: str, search_term: str) -> List[Dict[str, Any]]:
        """Search conversations by content"""
        return self.conversation_storage.search_conversations(document_id, search_term)
    
    def get_conversation_stats(self, document_id: str) -> Dict[str, Any]:
        """Get conversation statistics for a document"""
        return self.conversation_storage.get_conversation_stats(document_id)
    
    def export_conversations(self, document_id: str, format_type: str = "json") -> str:
        """Export conversations in different formats"""
        return self.conversation_storage.export_conversations(document_id, format_type)
    
    def get_conversations_with_charts(self, document_id: str) -> List[Dict[str, Any]]:
        """Get conversations that include chart data"""
        return self.conversation_storage.get_conversations_with_charts(document_id)
    
    def get_chart_types_used(self, document_id: str) -> Dict[str, int]:
        """Get statistics on chart types used"""
        return self.conversation_storage.get_chart_types_used(document_id)