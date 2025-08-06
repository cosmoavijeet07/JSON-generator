import numpy as np
from typing import List, Dict, Any, Optional
import openai
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import pickle
from pathlib import Path

class EmbeddingService:
    def __init__(self, cache_dir: str = "embeddings_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.embeddings_cache = {}
        self.client = openai.OpenAI()
    
    def get_embedding(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """Get embedding for text with caching"""
        # Create cache key
        cache_key = hashlib.md5(f"{text}:{model}".encode()).hexdigest()
        
        # Check cache
        if cache_key in self.embeddings_cache:
            return self.embeddings_cache[cache_key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                embedding = pickle.load(f)
                self.embeddings_cache[cache_key] = embedding
                return embedding
        
        # Generate embedding
        response = self.client.embeddings.create(
            input=text,
            model=model
        )
        embedding = response.data[0].embedding
        
        # Cache
        self.embeddings_cache[cache_key] = embedding
        with open(cache_file, "wb") as f:
            pickle.dump(embedding, f)
        
        return embedding
    
    def get_batch_embeddings(self, texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
        """Get embeddings for multiple texts"""
        embeddings = []
        
        # Process in batches (OpenAI has limits)
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = [self.get_embedding(text, model) for text in batch]
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between embeddings"""
        return cosine_similarity([embedding1], [embedding2])[0][0]
    
    def find_similar_chunks(self, query_embedding: List[float], 
                          chunk_embeddings: List[List[float]], 
                          top_k: int = 5) -> List[int]:
        """Find most similar chunks to query"""
        similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return top_indices.tolist()
    
    def create_document_embedding(self, text: str, chunk_size: int = 1000) -> Dict[str, Any]:
        """Create embeddings for document chunks"""
        # Simple chunking (can be improved)
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        embeddings = self.get_batch_embeddings(chunks)
        
        return {
            "chunks": chunks,
            "embeddings": embeddings,
            "chunk_size": chunk_size,
            "num_chunks": len(chunks)
        }