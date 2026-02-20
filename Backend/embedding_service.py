import google.generativeai as genai
from typing import List, Dict, Any
import numpy as np
import asyncio
import time
import os
from dotenv import load_dotenv

# Try to import OpenAI
try:
    from openai import AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("WARNING: OpenAI package not installed. Install with: pip install openai")

# Load environment variables
load_dotenv()

class EmbeddingService:
    def __init__(self, provider: str = None):
        """
        Initialize embedding service with specified provider
        
        Args:
            provider: 'gemini', 'azure_openai', or None (uses DEFAULT_EMBEDDING_PROVIDER from .env)
        """
        # Determine which provider to use
        self.provider = provider or os.getenv("DEFAULT_EMBEDDING_PROVIDER", "gemini")
        print(f"INFO: Initializing embedding service with provider: {self.provider}")
        
        # Configure Gemini API
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.use_gemini = gemini_api_key is not None and self.provider == "gemini"
        
        if self.use_gemini:
            genai.configure(api_key=gemini_api_key)
            print("INFO: Gemini API configured for embeddings.")
        
        # Configure Azure OpenAI
        self.use_azure_openai = False
        self.azure_client = None
        if OPENAI_AVAILABLE and self.provider == "azure_openai":
            azure_endpoint = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
            azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
            self.azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
            
            if azure_endpoint and azure_api_key:
                try:
                    self.azure_client = AzureOpenAI(
                        azure_endpoint=azure_endpoint,
                        api_key=azure_api_key,
                        api_version="2024-02-01"
                    )
                    self.use_azure_openai = True
                    print(f"INFO: Azure OpenAI configured for embeddings (deployment: {self.azure_deployment}).")
                except Exception as e:
                    print(f"WARNING: Failed to configure Azure OpenAI: {str(e)}")
            else:
                print("WARNING: Azure OpenAI credentials not found in .env")
        
        # Initialize local model as backup
        self.local_model = None
        self._init_local_model()
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
        
        # Embedding dimensions
        self.gemini_dimension = 3072  # Gemini embedding dimension
        self.azure_dimension = 1536   # Azure OpenAI text-embedding-3-small dimension
        self.local_dimension = 384    # Local model dimension
    
    def set_provider(self, provider: str):
        """Change the embedding provider dynamically"""
        if provider not in ["gemini", "azure_openai"]:
            raise ValueError(f"Invalid provider: {provider}. Must be 'gemini' or 'azure_openai'")
        
        print(f"INFO: Switching embedding provider to: {provider}")
        self.provider = provider
        self.use_gemini = provider == "gemini" and os.getenv("GEMINI_API_KEY") is not None
        self.use_azure_openai = provider == "azure_openai" and self.azure_client is not None
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[List[float]]:
        """Generate embeddings for text chunks using configured provider with fallback"""
        embeddings = []
        
        for chunk in chunks:
            text = chunk['text']
            
            # Try primary provider first
            if self.provider == "azure_openai":
                embedding = self._get_azure_openai_embedding(text)
            else:
                embedding = self._get_gemini_embedding(text)
            
            # Fallback to alternative provider
            if embedding is None:
                if self.provider == "azure_openai":
                    print("Azure OpenAI failed, trying Gemini...")
                    embedding = self._get_gemini_embedding(text)
                else:
                    print("Gemini failed, trying Azure OpenAI...")
                    embedding = self._get_azure_openai_embedding(text)
            
            # Fallback to local model
            if embedding is None and self.local_model:
                embedding = self._get_local_embedding(text)
            elif embedding is None:
                # Simple fallback - create a basic hash-based embedding
                embedding = self._create_simple_embedding(text)
            
            if embedding is not None:
                embeddings.append(embedding)
            else:
                # If all fail, create a zero vector (not ideal but prevents crashes)
                embeddings.append([0.0] * 384)  # Default dimension
        
        return embeddings
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a search query"""
        # Try primary provider first
        if self.provider == "azure_openai":
            embedding = self._get_azure_openai_embedding(query)
        else:
            embedding = self._get_gemini_embedding(query)
        
        # Fallback to alternative provider
        if embedding is None:
            if self.provider == "azure_openai":
                embedding = self._get_gemini_embedding(query)
            else:
                embedding = self._get_azure_openai_embedding(query)
        
        # Fallback to local model
        if embedding is None and self.local_model:
            embedding = self._get_local_embedding(query)
        elif embedding is None:
            # Simple fallback
            embedding = self._create_simple_embedding(query)
        
        return embedding if embedding is not None else [0.0] * 384
        return embeddings
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a search query"""
        # Try Gemini API first
        embedding = self._get_gemini_embedding(query)
        
        if embedding is None and self.local_model:
            # Fallback to local model
            embedding = self._get_local_embedding(query)
        elif embedding is None:
            # Simple fallback
            embedding = self._create_simple_embedding(query)
        
        return embedding if embedding is not None else [0.0] * 384
    
    def _get_gemini_embedding(self, text: str) -> List[float]:
        """Get embedding using Gemini API with rate limiting"""
        if not self.use_gemini:
            return None
            
        try:
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_request_interval:
                time.sleep(self.min_request_interval - time_since_last)
            
            # Truncate text if too long (Gemini has input limits)
            if len(text) > 2000:
                text = text[:2000]
            
            # Generate embedding using Gemini
            result = genai.embed_content(
                model="models/gemini-embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            
            self.last_request_time = time.time()
            return result['embedding']
            
        except Exception as e:
            if "quota" in str(e).lower() or "429" in str(e):
                print("WARNING: Gemini API quota exceeded, falling back to local model")
            else:
                print(f"Gemini embedding failed: {str(e)}")
            return None
    
    def _get_azure_openai_embedding(self, text: str) -> List[float]:
        """Get embedding using Azure OpenAI API"""
        if not self.use_azure_openai or not self.azure_client:
            return None
            
        try:
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_request_interval:
                time.sleep(self.min_request_interval - time_since_last)
            
            # Truncate text if too long
            if len(text) > 8000:
                text = text[:8000]
            
            # Generate embedding using Azure OpenAI
            response = self.azure_client.embeddings.create(
                model=self.azure_deployment,
                input=text
            )
            
            self.last_request_time = time.time()
            return response.data[0].embedding
            
        except Exception as e:
            if "quota" in str(e).lower() or "429" in str(e):
                print("WARNING: Azure OpenAI API quota exceeded")
            else:
                print(f"Azure OpenAI embedding failed: {str(e)}")
            return None
    
    def _init_local_model(self):
        """Initialize local sentence transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            import os
            
            # Try to load model with timeout and error handling
            self.local_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("INFO: Local sentence transformer model loaded successfully.")
        except ImportError:
            print("WARNING: sentence-transformers not installed. Will use fallback embeddings.")
            self.local_model = None
        except Exception as e:
            print(f"WARNING: Failed to load local model (network/download issue): {e}")
            print("INFO: Will use improved fallback embeddings instead.")
            self.local_model = None

    def _get_local_embedding(self, text: str) -> List[float]:
        """Get embedding using local sentence transformer"""
        try:
            if self.local_model is None:
                return None
            
            embedding = self.local_model.encode(text)
            # Pad to match Gemini dimension (768)
            if len(embedding) < self.gemini_dimension:
                padding = [0.0] * (self.gemini_dimension - len(embedding))
                embedding = np.concatenate([embedding, padding])
            elif len(embedding) > self.gemini_dimension:
                embedding = embedding[:self.gemini_dimension]
            
            return embedding.tolist()
            
        except Exception as e:
            print(f"Local embedding failed: {str(e)}")
            return None
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            magnitude1 = np.linalg.norm(vec1)
            magnitude2 = np.linalg.norm(vec2)
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            similarity = dot_product / (magnitude1 * magnitude2)
            return float(similarity)
            
        except Exception as e:
            print(f"Similarity calculation failed: {str(e)}")
            return 0.0
    
    async def batch_generate_embeddings(self, texts: List[str], batch_size: int = 5) -> List[List[float]]:
        """Generate embeddings in batches to avoid rate limits"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = []
            
            for text in batch:
                embedding = self.generate_query_embedding(text)
                batch_embeddings.append(embedding)
            
            embeddings.extend(batch_embeddings)
            
            # Wait between batches to avoid rate limits
            if i + batch_size < len(texts):
                await asyncio.sleep(2)
        
        return embeddings
    
    def get_embedding_stats(self, embeddings: List[List[float]]) -> Dict[str, Any]:
        """Get statistics about the embeddings"""
        if not embeddings:
            return {"count": 0}
        
        dimensions = len(embeddings[0])
        
        # Convert to numpy for calculations
        embedding_matrix = np.array(embeddings)
        
        return {
            "count": len(embeddings),
            "dimensions": dimensions,
            "mean_magnitude": float(np.mean(np.linalg.norm(embedding_matrix, axis=1))),
            "std_magnitude": float(np.std(np.linalg.norm(embedding_matrix, axis=1))),
            "min_value": float(np.min(embedding_matrix)),
            "max_value": float(np.max(embedding_matrix))
        }
    
    def _create_simple_embedding(self, text: str) -> List[float]:
        """Create a sophisticated TF-IDF-like embedding as fallback"""
        import re
        from collections import Counter
        import math
        
        # Enhanced tokenization
        text_clean = re.sub(r'[^\w\s]', ' ', text.lower())
        words = [word for word in text_clean.split() if len(word) > 2]
        
        # Expanded domain-specific vocabulary
        ml_vocabulary = [
            # Core ML terms
            'machine', 'learning', 'artificial', 'intelligence', 'algorithm', 'model', 'data',
            'neural', 'network', 'deep', 'training', 'supervised', 'unsupervised', 'reinforcement',
            'classification', 'regression', 'clustering', 'prediction', 'feature', 'dataset',
            'optimization', 'gradient', 'descent', 'backpropagation', 'overfitting', 'validation',
            'test', 'accuracy', 'precision', 'recall', 'cross', 'validation', 'bias', 'variance',
            
            # Technical terms
            'computer', 'science', 'technology', 'system', 'information', 'processing', 'analysis',
            'statistical', 'probability', 'distribution', 'function', 'linear', 'logistic',
            'decision', 'tree', 'forest', 'support', 'vector', 'bayes', 'naive', 'knn',
            
            # Application domains
            'vision', 'language', 'speech', 'recognition', 'recommendation', 'autonomous',
            'medical', 'diagnosis', 'image', 'text', 'nlp', 'computer', 'robotics',
            
            # Common words for context
            'what', 'how', 'why', 'when', 'where', 'definition', 'example', 'application',
            'method', 'technique', 'approach', 'problem', 'solution', 'result', 'performance'
        ]
        
        # Calculate word frequencies and TF scores
        word_counts = Counter(words)
        total_words = len(words)
        
        # Create sophisticated embedding
        embedding = []
        
        # 1. Vocabulary-based features (300 dimensions)
        for word in ml_vocabulary[:300]:
            if word in word_counts:
                # TF score with log normalization
                tf = word_counts[word] / total_words
                tf_log = math.log(1 + tf * 100)  # Scaled log
                embedding.append(tf_log)
            else:
                embedding.append(0.0)
        
        # 2. N-gram features (200 dimensions)
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
        ml_bigrams = [
            'machine_learning', 'artificial_intelligence', 'neural_network', 'deep_learning',
            'supervised_learning', 'unsupervised_learning', 'reinforcement_learning',
            'decision_tree', 'random_forest', 'support_vector', 'naive_bayes',
            'linear_regression', 'logistic_regression', 'gradient_descent',
            'computer_vision', 'natural_language', 'speech_recognition'
        ]
        
        bigram_counts = Counter(bigrams)
        for bigram in ml_bigrams:
            if bigram in bigram_counts:
                embedding.append(min(bigram_counts[bigram] / len(bigrams), 1.0))
            else:
                embedding.append(0.0)
        
        # Pad bigram section to 200 dimensions
        while len(embedding) < 500:
            embedding.append(0.0)
        
        # 3. Statistical and structural features (268 dimensions)
        structural_features = [
            len(text) / 1000.0,  # Text length
            len(words) / 100.0,  # Word count
            len(set(words)) / max(len(words), 1),  # Vocabulary diversity
            sum(len(word) for word in words) / max(len(words), 1) / 10.0,  # Avg word length
            text.count('?') / max(len(text), 1) * 100,  # Question density
            text.count('.') / max(len(text), 1) * 100,  # Sentence density
            len([w for w in words if w in ml_vocabulary]) / max(len(words), 1),  # ML term density
        ]
        
        embedding.extend(structural_features)
        
        # Pad to exactly 768 dimensions
        while len(embedding) < self.gemini_dimension:
            embedding.append(0.0)
        
        # Normalize the embedding to unit length for better similarity calculation
        magnitude = math.sqrt(sum(x*x for x in embedding))
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return embedding[:self.gemini_dimension]