import numpy as np
import hashlib
from typing import List, Union, Optional, Tuple
import json
import pickle
import os

# Try to import sentence-transformers, fall back to simple implementation if not available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Using fallback embedding method.")

class EmbeddingManager:
    """Manages text embeddings using sentence-transformers or fallback method"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', cache_dir: str = './embeddings_cache'):
        """
        Initialize embedding manager
        
        Args:
            model_name: Name of sentence-transformer model to use
            cache_dir: Directory to cache embeddings
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                self.use_transformer = True
            except Exception as e:
                print(f"Error loading sentence-transformer model: {e}")
                self.use_transformer = False
                self.embedding_dim = 384  # Default dimension
        else:
            self.use_transformer = False
            self.embedding_dim = 384  # Default dimension for fallback
        
        self.embedding_cache = {}
    
    def create_embeddings(self, text: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        Create embeddings for text using sentence-transformers or fallback
        
        Args:
            text: Single text or list of texts to embed
            batch_size: Batch size for encoding (if using transformer)
        
        Returns:
            Numpy array of embeddings
        """
        # Ensure text is a list
        if isinstance(text, str):
            texts = [text]
            single_input = True
        else:
            texts = text
            single_input = False
        
        # Check cache
        embeddings = []
        texts_to_encode = []
        cache_indices = []
        
        for i, t in enumerate(texts):
            cache_key = self._get_cache_key(t)
            if cache_key in self.embedding_cache:
                embeddings.append(self.embedding_cache[cache_key])
            else:
                texts_to_encode.append(t)
                cache_indices.append(i)
        
        # Encode uncached texts
        if texts_to_encode:
            if self.use_transformer:
                new_embeddings = self._encode_with_transformer(texts_to_encode, batch_size)
            else:
                new_embeddings = self._encode_fallback(texts_to_encode)
            
            # Add to cache
            for t, emb in zip(texts_to_encode, new_embeddings):
                cache_key = self._get_cache_key(t)
                self.embedding_cache[cache_key] = emb
            
            # Merge with cached embeddings
            new_emb_iter = iter(new_embeddings)
            final_embeddings = []
            for i in range(len(texts)):
                if i in cache_indices:
                    final_embeddings.append(next(new_emb_iter))
                else:
                    final_embeddings.append(embeddings.pop(0))
            
            embeddings = np.array(final_embeddings)
        else:
            embeddings = np.array(embeddings)
        
        # Return single embedding if single input
        if single_input:
            return embeddings[0]
        
        return embeddings
    
    def _encode_with_transformer(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Encode texts using sentence-transformer model"""
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    
    def _encode_fallback(self, texts: List[str]) -> np.ndarray:
        """Fallback encoding method using TF-IDF-like features"""
        embeddings = []
        
        for text in texts:
            # Create feature vector
            features = []
            
            # Basic text statistics
            features.extend([
                len(text) / 10000,  # Normalized length
                text.count(' ') / 1000,  # Word count approximation
                text.count('.') / 100,  # Sentence count
                text.count('\n') / 50,  # Line count
            ])
            
            # Character frequency features
            text_lower = text.lower()
            for char in 'etaoinshrdlcumwfgypbvkjxqz':
                features.append(text_lower.count(char) / max(len(text), 1))
            
            # Common word features
            common_words = ['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 
                          'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 
                          'do', 'at', 'this', 'but', 'his', 'by', 'from']
            
            for word in common_words:
                features.append(text_lower.count(word) / 100)
            
            # Pad or truncate to embedding dimension
            if len(features) < self.embedding_dim:
                features.extend([0.0] * (self.embedding_dim - len(features)))
            else:
                features = features[:self.embedding_dim]
            
            # Normalize
            features = np.array(features)
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            
            embeddings.append(features)
        
        return np.array(embeddings)
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
        
        Returns:
            Cosine similarity score (-1 to 1)
        """
        # Ensure 1D arrays
        if embedding1.ndim > 1:
            embedding1 = embedding1.flatten()
        if embedding2.ndim > 1:
            embedding2 = embedding2.flatten()
        
        # Calculate cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def find_most_similar(
        self, 
        query_embedding: np.ndarray, 
        candidate_embeddings: np.ndarray,
        top_k: int = 5
    ) -> Tuple[List[int], List[float]]:
        """
        Find the most similar embeddings to a query
        
        Args:
            query_embedding: Query embedding
            candidate_embeddings: Array of candidate embeddings
            top_k: Number of top results to return
        
        Returns:
            Tuple of (indices, similarities) for top k matches
        """
        similarities = []
        
        for i, candidate in enumerate(candidate_embeddings):
            sim = self.calculate_similarity(query_embedding, candidate)
            similarities.append((i, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top k
        top_k = min(top_k, len(similarities))
        indices = [s[0] for s in similarities[:top_k]]
        scores = [s[1] for s in similarities[:top_k]]
        
        return indices, scores
    
    def semantic_search(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[Tuple[int, str, float]]:
        """
        Perform semantic search on documents
        
        Args:
            query: Search query
            documents: List of documents to search
            top_k: Number of top results
        
        Returns:
            List of tuples (index, document, similarity_score)
        """
        # Encode query and documents
        query_embedding = self.create_embeddings(query)
        doc_embeddings = self.create_embeddings(documents)
        
        # Find most similar
        indices, scores = self.find_most_similar(query_embedding, doc_embeddings, top_k)
        
        # Return results
        results = []
        for idx, score in zip(indices, scores):
            results.append((idx, documents[idx], score))
        
        return results
    
    def save_embeddings(self, embeddings: np.ndarray, filepath: str):
        """Save embeddings to file"""
        np.save(filepath, embeddings)
    
    def load_embeddings(self, filepath: str) -> np.ndarray:
        """Load embeddings from file"""
        return np.load(filepath)
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self.embedding_cache.clear()
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.embedding_dim

# Initialize global instance
embedding_manager = EmbeddingManager()