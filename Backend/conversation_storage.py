import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import uuid

class ConversationStorage:
    def __init__(self, storage_dir: str = "conversations"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
    
    def _get_conversation_file(self, document_id: str) -> str:
        """Get the file path for a document's conversations"""
        return os.path.join(self.storage_dir, f"{document_id}_conversations.json")
    
    def save_conversation(self, document_id: str, query: str, response: str, 
                         sources: List[Dict] = None, query_type: str = "general",
                         confidence: float = 0.0, chart_data: Dict = None) -> str:
        """Save a conversation exchange"""
        conversation_id = str(uuid.uuid4())
        
        conversation_entry = {
            "id": conversation_id,
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "query_type": query_type,
            "confidence": confidence,
            "sources": sources or [],
            "chart_data": chart_data,
            "has_chart": chart_data is not None,
            "chart_type": chart_data.get("type") if chart_data else None
        }
        
        # Load existing conversations
        conversations = self.load_conversations(document_id)
        conversations.append(conversation_entry)
        
        # Save back to file
        file_path = self._get_conversation_file(document_id)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, indent=2, ensure_ascii=False)
        
        return conversation_id
    
    def load_conversations(self, document_id: str) -> List[Dict[str, Any]]:
        """Load all conversations for a document"""
        file_path = self._get_conversation_file(document_id)
        
        if not os.path.exists(file_path):
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def get_recent_conversations(self, document_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent conversations for context"""
        conversations = self.load_conversations(document_id)
        return conversations[-limit:] if conversations else []
    
    def clear_conversations(self, document_id: str) -> bool:
        """Clear all conversations for a document"""
        file_path = self._get_conversation_file(document_id)
        
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                return True
            except OSError:
                return False
        return True
    
    def delete_conversation(self, document_id: str, conversation_id: str) -> bool:
        """Delete a specific conversation"""
        conversations = self.load_conversations(document_id)
        
        # Filter out the conversation to delete
        updated_conversations = [
            conv for conv in conversations 
            if conv.get("id") != conversation_id
        ]
        
        if len(updated_conversations) != len(conversations):
            # Save updated conversations
            file_path = self._get_conversation_file(document_id)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(updated_conversations, f, indent=2, ensure_ascii=False)
            return True
        
        return False
    
    def search_conversations(self, document_id: str, search_term: str) -> List[Dict[str, Any]]:
        """Search conversations by query or response content"""
        conversations = self.load_conversations(document_id)
        search_term_lower = search_term.lower()
        
        matching_conversations = []
        for conv in conversations:
            if (search_term_lower in conv.get("query", "").lower() or 
                search_term_lower in conv.get("response", "").lower()):
                matching_conversations.append(conv)
        
        return matching_conversations
    
    def get_conversation_stats(self, document_id: str) -> Dict[str, Any]:
        """Get statistics about conversations for a document"""
        conversations = self.load_conversations(document_id)
        
        if not conversations:
            return {
                "total_conversations": 0,
                "query_types": {},
                "average_confidence": 0.0,
                "first_conversation": None,
                "last_conversation": None
            }
        
        # Calculate query type distribution
        query_types = {}
        total_confidence = 0
        
        for conv in conversations:
            query_type = conv.get("query_type", "general")
            query_types[query_type] = query_types.get(query_type, 0) + 1
            total_confidence += conv.get("confidence", 0)
        
        return {
            "total_conversations": len(conversations),
            "query_types": query_types,
            "average_confidence": total_confidence / len(conversations) if conversations else 0,
            "first_conversation": conversations[0]["timestamp"] if conversations else None,
            "last_conversation": conversations[-1]["timestamp"] if conversations else None
        }
    
    def get_conversations_with_charts(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all conversations that include chart data"""
        conversations = self.load_conversations(document_id)
        return [conv for conv in conversations if conv.get("has_chart", False)]
    
    def get_chart_types_used(self, document_id: str) -> Dict[str, int]:
        """Get statistics on chart types used in conversations"""
        conversations = self.load_conversations(document_id)
        chart_types = {}
        
        for conv in conversations:
            if conv.get("has_chart", False):
                chart_type = conv.get("chart_type", "unknown")
                chart_types[chart_type] = chart_types.get(chart_type, 0) + 1
        
        return chart_types
    
    def export_conversations(self, document_id: str, format_type: str = "json") -> str:
        """Export conversations in different formats"""
        conversations = self.load_conversations(document_id)
        
        if format_type == "json":
            return json.dumps(conversations, indent=2, ensure_ascii=False)
        
        elif format_type == "text":
            text_export = f"Conversation History for Document: {document_id}\n"
            text_export += "=" * 50 + "\n\n"
            
            for i, conv in enumerate(conversations, 1):
                text_export += f"Conversation {i} - {conv['timestamp']}\n"
                text_export += f"Query Type: {conv.get('query_type', 'general')}\n"
                text_export += f"Confidence: {conv.get('confidence', 0):.1f}%\n"
                if conv.get('has_chart', False):
                    text_export += f"Chart: {conv.get('chart_type', 'unknown')} chart included\n"
                text_export += f"Query: {conv['query']}\n"
                text_export += f"Response: {conv['response']}\n"
                text_export += "-" * 30 + "\n\n"
            
            return text_export
        
        else:
            raise ValueError(f"Unsupported format type: {format_type}")