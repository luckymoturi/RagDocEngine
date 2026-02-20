"""
BERT-based Question Answering Service
Uses transformer models for extractive QA from document context
"""

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from typing import Dict, Any, List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer, util
import asyncio


class BertQAService:
    """BERT-based extractive question answering service"""
    
    def __init__(self):
        # Use distilbert-base-cased-distilled-squad - better at extracting answers
        # Alternative: "bert-large-uncased-whole-word-masking-finetuned-squad" for higher accuracy
        self.model_name = "distilbert-base-cased-distilled-squad"
        self.tokenizer = None
        self.model = None
        self.qa_pipeline = None
        
        # Sentence transformer for semantic similarity comparison
        self.sentence_model = None
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize BERT QA and sentence transformer models"""
        try:
            print("Loading BERT QA model...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
            
            # Create QA pipeline for easier inference
            self.qa_pipeline = pipeline(
                "question-answering",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1  # CPU, use 0 for GPU
            )
            
            print("Loading sentence transformer for answer comparison...")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            print("BERT QA models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading BERT models: {str(e)}")
            raise
    
    async def answer_question(
        self, 
        question: str, 
        context_chunks: List[Dict[str, Any]],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Answer a question using BERT extractive QA
        
        Args:
            question: User's question
            context_chunks: List of document chunks with text
            top_k: Number of top answers to consider
            
        Returns:
            Dict with answer, confidence, and metadata
        """
        try:
            # Combine context from chunks
            context = self._prepare_context(context_chunks)
            
            if not context.strip():
                return {
                    "answer": "No relevant context found to answer the question.",
                    "confidence": 0.0,
                    "source": "bert",
                    "error": True
                }
            
            # Run BERT QA - handle in thread pool for async
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._run_qa(question, context, top_k)
            )
            
            return result
            
        except Exception as e:
            return {
                "answer": f"BERT QA error: {str(e)}",
                "confidence": 0.0,
                "source": "bert",
                "error": True
            }
    
    def _prepare_context(self, context_chunks: List[Dict[str, Any]], max_length: int = 4000) -> str:
        """Prepare context from chunks, respecting token limits"""
        context_parts = []
        total_length = 0
        
        for chunk in context_chunks:
            chunk_text = chunk.get("chunk_text", chunk.get("text", ""))
            if total_length + len(chunk_text) > max_length:
                # Add partial if there's room
                remaining = max_length - total_length
                if remaining > 100:
                    context_parts.append(chunk_text[:remaining])
                break
            context_parts.append(chunk_text)
            total_length += len(chunk_text)
        
        return "\n\n".join(context_parts)
    
    def _run_qa(self, question: str, context: str, top_k: int) -> Dict[str, Any]:
        """Run BERT QA inference"""
        try:
            # Split context into manageable chunks for BERT (512 token limit)
            context_windows = self._split_context_for_bert(context)
            
            all_answers = []
            
            for window in context_windows:
                try:
                    # Don't use handle_impossible_answer - always extract something
                    result = self.qa_pipeline(
                        question=question,
                        context=window,
                        top_k=top_k,
                        max_answer_len=200
                    )
                    
                    # Handle both single result and list of results
                    if isinstance(result, list):
                        all_answers.extend(result)
                    else:
                        all_answers.append(result)
                        
                except Exception as e:
                    print(f"Error processing context window: {str(e)}")
                    continue
            
            if not all_answers:
                return {
                    "answer": "Could not extract an answer from the document.",
                    "confidence": 0.0,
                    "source": "bert",
                    "model": self.model_name
                }
            
            # Sort by confidence and get best answer
            all_answers.sort(key=lambda x: x.get('score', 0), reverse=True)
            best_answer = all_answers[0]
            
            # Format the answer with context
            answer_text = best_answer.get('answer', '').strip()
            confidence = best_answer.get('score', 0.0)
            
            # Always return the extracted answer (don't say "no answer found")
            # Even low confidence answers can be useful
            if not answer_text:
                # If truly empty, try to get any answer from candidates
                for ans in all_answers:
                    if ans.get('answer', '').strip():
                        answer_text = ans.get('answer', '').strip()
                        confidence = ans.get('score', 0.0)
                        break
            
            if not answer_text:
                return {
                    "answer": "Could not extract a specific answer from the document.",
                    "confidence": 0.0,
                    "source": "bert",
                    "model": self.model_name
                }
            
            # Expand answer with surrounding context for better readability
            expanded_answer = self._expand_answer(answer_text, context, question)
            
            return {
                "answer": expanded_answer,
                "confidence": float(confidence),
                "source": "bert",
                "model": self.model_name,
                "raw_answer": answer_text,
                "all_candidates": [
                    {"answer": a.get('answer', ''), "score": float(a.get('score', 0))}
                    for a in all_answers[:5]
                ]
            }
            
        except Exception as e:
            return {
                "answer": f"Error during QA: {str(e)}",
                "confidence": 0.0,
                "source": "bert",
                "error": True
            }
    
    def _split_context_for_bert(self, context: str, max_tokens: int = 450) -> List[str]:
        """Split context into windows that fit BERT's token limit"""
        # Approximate: 1 token ≈ 4 characters, but be conservative
        max_chars = max_tokens * 3
        
        if len(context) <= max_chars:
            return [context]
        
        windows = []
        sentences = context.replace('\n', ' ').split('. ')
        
        current_window = ""
        for sentence in sentences:
            if len(current_window) + len(sentence) + 2 <= max_chars:
                current_window += sentence + ". "
            else:
                if current_window:
                    windows.append(current_window.strip())
                current_window = sentence + ". "
        
        if current_window:
            windows.append(current_window.strip())
        
        # Also create overlapping windows for better coverage
        if len(windows) > 1:
            overlapping = []
            for i in range(len(windows) - 1):
                # Combine adjacent windows if they fit
                combined = windows[i][-500:] + " " + windows[i+1][:500]
                if len(combined) <= max_chars:
                    overlapping.append(combined)
            windows.extend(overlapping)
        
        return windows if windows else [context[:max_chars]]
    
    def _expand_answer(self, answer: str, context: str, question: str) -> str:
        """Expand the extracted answer with surrounding context"""
        if len(answer) > 150:
            return answer
        
        # Find answer in context and get surrounding text
        answer_lower = answer.lower()
        context_lower = context.lower()
        
        pos = context_lower.find(answer_lower)
        if pos == -1:
            return answer
        
        # Get surrounding context (1-2 sentences)
        start = max(0, context.rfind('.', 0, pos) + 1)
        end = context.find('.', pos + len(answer))
        if end == -1:
            end = min(len(context), pos + len(answer) + 200)
        else:
            # Get one more sentence for context
            next_end = context.find('.', end + 1)
            if next_end != -1 and next_end - start < 400:
                end = next_end + 1
            else:
                end = end + 1
        
        expanded = context[start:end].strip()
        
        # If expanded is too long, just return original
        if len(expanded) > 500:
            return answer
        
        return expanded if expanded else answer


class AnswerComparator:
    """Compare RAG and BERT answers to select the best one"""
    
    def __init__(self, sentence_model: SentenceTransformer = None):
        self.sentence_model = sentence_model or SentenceTransformer('all-MiniLM-L6-v2')
    
    async def compare_and_select(
        self,
        question: str,
        rag_answer: Dict[str, Any],
        bert_answer: Dict[str, Any],
        context_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare RAG and BERT answers and select the best one
        
        Criteria:
        1. Relevance to question (semantic similarity)
        2. Grounding in source documents
        3. Confidence scores
        4. Answer completeness
        """
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._compare_answers(question, rag_answer, bert_answer, context_chunks)
            )
            return result
        except Exception as e:
            # Default to RAG on error
            return {
                "selected": "rag",
                "reason": f"Comparison error: {str(e)}",
                "best_answer": rag_answer.get("answer", ""),
                "scores": {"rag": 0.5, "bert": 0.5}
            }
    
    def _compare_answers(
        self,
        question: str,
        rag_answer: Dict[str, Any],
        bert_answer: Dict[str, Any],
        context_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform the actual comparison"""
        
        rag_text = rag_answer.get("answer", "") if isinstance(rag_answer, dict) else str(rag_answer)
        bert_text = bert_answer.get("answer", "") if isinstance(bert_answer, dict) else str(bert_answer)
        
        # Handle empty answers
        if not rag_text.strip() and not bert_text.strip():
            return {
                "selected": "none",
                "reason": "Both answers are empty",
                "best_answer": "Unable to generate an answer.",
                "scores": {"rag": 0, "bert": 0}
            }
        
        if not bert_text.strip() or bert_answer.get("error"):
            return {
                "selected": "rag",
                "reason": "BERT answer unavailable",
                "best_answer": rag_text,
                "scores": {"rag": 1.0, "bert": 0}
            }
        
        if not rag_text.strip():
            return {
                "selected": "bert",
                "reason": "RAG answer unavailable",
                "best_answer": bert_text,
                "scores": {"rag": 0, "bert": 1.0}
            }
        
        # Calculate scores for each answer
        rag_score = self._calculate_answer_score(question, rag_text, context_chunks, rag_answer)
        bert_score = self._calculate_answer_score(question, bert_text, context_chunks, bert_answer)
        
        # Determine winner
        if rag_score > bert_score:
            selected = "rag"
            best_answer = rag_text
            reason = self._generate_reason(rag_score, bert_score, "RAG")
        elif bert_score > rag_score:
            selected = "bert"
            best_answer = bert_text
            reason = self._generate_reason(bert_score, rag_score, "BERT")
        else:
            # Tie - prefer RAG as it's more comprehensive
            selected = "rag"
            best_answer = rag_text
            reason = "Scores tied, preferring RAG for comprehensiveness"
        
        return {
            "selected": selected,
            "reason": reason,
            "best_answer": best_answer,
            "scores": {
                "rag": round(rag_score, 3),
                "bert": round(bert_score, 3)
            },
            "rag_answer": rag_text,
            "bert_answer": bert_text,
            "comparison_details": {
                "rag_metrics": self._get_detailed_metrics(question, rag_text, context_chunks),
                "bert_metrics": self._get_detailed_metrics(question, bert_text, context_chunks)
            }
        }
    
    def _calculate_answer_score(
        self,
        question: str,
        answer: str,
        context_chunks: List[Dict[str, Any]],
        answer_meta: Dict[str, Any]
    ) -> float:
        """Calculate a comprehensive score for an answer"""
        
        # 1. Question-Answer Relevance (40% weight)
        qa_relevance = self._semantic_similarity(question, answer)
        
        # 2. Answer-Context Grounding (30% weight)
        context_text = " ".join([c.get("chunk_text", c.get("text", "")) for c in context_chunks[:3]])
        grounding = self._semantic_similarity(answer, context_text) if context_text else 0.5
        
        # 3. Confidence Score (20% weight)
        confidence = answer_meta.get("confidence", 0.5)
        if isinstance(confidence, (int, float)):
            confidence = min(1.0, max(0.0, float(confidence)))
        else:
            confidence = 0.5
        
        # 4. Answer Quality (10% weight)
        quality = self._assess_answer_quality(answer)
        
        # Weighted combination
        total_score = (
            qa_relevance * 0.40 +
            grounding * 0.30 +
            confidence * 0.20 +
            quality * 0.10
        )
        
        return total_score
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        try:
            if not text1.strip() or not text2.strip():
                return 0.0
            
            embeddings = self.sentence_model.encode([text1, text2])
            similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
            return max(0.0, min(1.0, similarity))
        except:
            return 0.5
    
    def _assess_answer_quality(self, answer: str) -> float:
        """Assess answer quality based on various factors"""
        if not answer:
            return 0.0
        
        score = 0.5  # Base score
        
        # Length check (not too short, not too long)
        length = len(answer)
        if 50 <= length <= 1000:
            score += 0.2
        elif 20 <= length < 50 or 1000 < length <= 2000:
            score += 0.1
        
        # Contains complete sentences
        if answer.strip().endswith(('.', '!', '?')):
            score += 0.1
        
        # Not an error message
        error_indicators = ['error', 'unable', 'cannot', 'failed', "don't know", "no information"]
        if not any(ind in answer.lower() for ind in error_indicators):
            score += 0.2
        
        return min(1.0, score)
    
    def _get_detailed_metrics(
        self,
        question: str,
        answer: str,
        context_chunks: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Get detailed metrics for an answer"""
        context_text = " ".join([c.get("chunk_text", c.get("text", "")) for c in context_chunks[:3]])
        
        return {
            "question_relevance": round(self._semantic_similarity(question, answer), 3),
            "context_grounding": round(self._semantic_similarity(answer, context_text), 3) if context_text else 0.5,
            "answer_quality": round(self._assess_answer_quality(answer), 3),
            "answer_length": len(answer)
        }
    
    def _generate_reason(self, winner_score: float, loser_score: float, winner_name: str) -> str:
        """Generate human-readable reason for selection"""
        diff = winner_score - loser_score
        
        if diff > 0.3:
            return f"{winner_name} significantly more relevant and grounded"
        elif diff > 0.15:
            return f"{winner_name} better matches question and context"
        else:
            return f"{winner_name} slightly better overall score"
