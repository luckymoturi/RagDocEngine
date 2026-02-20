"""
Smart Insights Generator
Automatically generates intelligent insights from documents
"""

import google.generativeai as genai
from typing import List, Dict, Any
import json
import os
from dotenv import load_dotenv

load_dotenv()


class SmartInsights:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
    
    async def generate_document_insights(self, document_chunks: List[str], metadata: Dict) -> Dict[str, Any]:
        """
        Generate comprehensive insights from a document
        Returns: Key insights, questions, connections, and recommendations
        """
        
        # Combine chunks for analysis
        combined_text = "\n\n".join(document_chunks[:10])  # Use first 10 chunks
        
        prompt = f"""
        Analyze this document and provide intelligent insights in JSON format.
        
        Document: {metadata.get('document_name', 'Unknown')}
        Content:
        {combined_text[:5000]}
        
        Generate insights in this EXACT JSON format:
        {{
            "key_insights": [
                "Insight 1",
                "Insight 2",
                "Insight 3"
            ],
            "surprising_facts": [
                "Fact 1",
                "Fact 2"
            ],
            "questions_to_explore": [
                "Question 1?",
                "Question 2?",
                "Question 3?"
            ],
            "related_topics": [
                "Topic 1",
                "Topic 2",
                "Topic 3"
            ],
            "practical_applications": [
                "Application 1",
                "Application 2"
            ],
            "complexity_level": "Beginner|Intermediate|Advanced",
            "estimated_read_time": "X minutes",
            "document_type": "Research|Technical|Business|Educational|Other",
            "key_entities": ["Entity1", "Entity2", "Entity3"],
            "sentiment": "Positive|Neutral|Negative|Mixed"
        }}
        
        Provide ONLY the JSON, no additional text.
        """
        
        try:
            response = self.model.generate_content(prompt)
            insights = json.loads(response.text)
            return insights
        except Exception as e:
            print(f"Error generating insights: {str(e)}")
            return self._get_default_insights()
    
    async def generate_smart_questions(self, document_chunks: List[str]) -> List[Dict[str, str]]:
        """
        Generate smart, contextual questions users might want to ask
        """
        
        combined_text = "\n\n".join(document_chunks[:5])
        
        prompt = f"""
        Based on this document content, generate 5 smart questions that users would want to ask.
        Make them specific, insightful, and varied in type.
        
        Content:
        {combined_text[:3000]}
        
        Return ONLY a JSON array:
        [
            {{"question": "Question 1?", "type": "factual|analytical|comparative|exploratory"}},
            {{"question": "Question 2?", "type": "factual|analytical|comparative|exploratory"}},
            ...
        ]
        """
        
        try:
            response = self.model.generate_content(prompt)
            questions = json.loads(response.text)
            return questions
        except Exception as e:
            print(f"Error generating questions: {str(e)}")
            return [
                {"question": "What is the main topic of this document?", "type": "factual"},
                {"question": "What are the key findings?", "type": "analytical"},
                {"question": "How does this relate to current trends?", "type": "exploratory"}
            ]
    
    async def generate_document_connections(
        self, 
        current_doc_chunks: List[str],
        other_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Find connections between current document and other documents
        """
        
        if not other_docs:
            return []
        
        current_summary = "\n".join(current_doc_chunks[:3])[:1000]
        other_summaries = "\n\n".join([
            f"Doc {i+1} ({doc['name']}): {doc['summary'][:200]}"
            for i, doc in enumerate(other_docs[:5])
        ])
        
        prompt = f"""
        Find connections between the current document and other documents.
        
        Current Document:
        {current_summary}
        
        Other Documents:
        {other_summaries}
        
        Return JSON array of connections:
        [
            {{
                "document_name": "Doc name",
                "connection_type": "Similar Topic|Contradicts|Extends|References",
                "description": "Brief description of connection",
                "strength": "Strong|Medium|Weak"
            }}
        ]
        
        Return ONLY the JSON array, maximum 3 connections.
        """
        
        try:
            response = self.model.generate_content(prompt)
            connections = json.loads(response.text)
            return connections[:3]
        except Exception as e:
            print(f"Error finding connections: {str(e)}")
            return []
    
    async def generate_learning_path(self, document_chunks: List[str]) -> Dict[str, Any]:
        """
        Generate a learning path based on document content
        """
        
        combined_text = "\n\n".join(document_chunks[:5])
        
        prompt = f"""
        Create a learning path based on this document content.
        
        Content:
        {combined_text[:3000]}
        
        Return JSON:
        {{
            "prerequisites": ["Concept 1", "Concept 2"],
            "learning_steps": [
                {{"step": 1, "title": "Step title", "description": "What to learn"}},
                {{"step": 2, "title": "Step title", "description": "What to learn"}}
            ],
            "next_topics": ["Topic 1", "Topic 2"],
            "difficulty_progression": "Easy → Medium → Hard"
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            learning_path = json.loads(response.text)
            return learning_path
        except Exception as e:
            print(f"Error generating learning path: {str(e)}")
            return {
                "prerequisites": [],
                "learning_steps": [],
                "next_topics": [],
                "difficulty_progression": "Not available"
            }
    
    async def generate_quick_facts(self, document_chunks: List[str]) -> List[str]:
        """
        Extract quick, interesting facts from the document
        """
        
        combined_text = "\n\n".join(document_chunks[:5])
        
        prompt = f"""
        Extract 5 quick, interesting facts from this document.
        Make them concise and engaging.
        
        Content:
        {combined_text[:3000]}
        
        Return as JSON array of strings:
        ["Fact 1", "Fact 2", "Fact 3", "Fact 4", "Fact 5"]
        """
        
        try:
            response = self.model.generate_content(prompt)
            facts = json.loads(response.text)
            return facts[:5]
        except Exception as e:
            print(f"Error generating facts: {str(e)}")
            return []
    
    def _get_default_insights(self) -> Dict[str, Any]:
        """Return default insights structure"""
        return {
            "key_insights": ["Document uploaded successfully"],
            "surprising_facts": [],
            "questions_to_explore": [
                "What is the main topic?",
                "What are the key points?",
                "How can I apply this information?"
            ],
            "related_topics": [],
            "practical_applications": [],
            "complexity_level": "Unknown",
            "estimated_read_time": "Unknown",
            "document_type": "Unknown",
            "key_entities": [],
            "sentiment": "Neutral"
        }
