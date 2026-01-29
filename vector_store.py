import os
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss

class BrainTumorVectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", index_path: str = "vector_index.faiss"):
        self.model_name = model_name
        self.index_path = index_path
        self.embedding_model = None
        self.index = None
        self.documents = []
        self.metadata = []
        
    def _load_embedding_model(self):
        """Load the sentence transformer model"""
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(self.model_name)
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of texts"""
        self._load_embedding_model()
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        return embeddings.astype('float32')
    
    def add_documents(self, documents: List[Dict]):
        """Add documents to the vector store"""
        self.documents.extend(documents)
        
        # Create text representations for embedding
        texts = []
        for doc in documents:
            # Combine title, content, and keywords for better representation
            text = f"{doc['title']} {doc['content']} {' '.join(doc['keywords'])}"
            texts.append(text)
        
        # Create embeddings
        embeddings = self.create_embeddings(texts)
        
        # Initialize FAISS index if not exists
        if self.index is None:
            embedding_dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(embedding_dim)  # Inner Product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        start_idx = len(self.metadata)
        self.index.add(embeddings)
        
        # Store metadata
        for i, doc in enumerate(documents):
            self.metadata.append({
                'doc_id': doc.get('id', f'doc_{start_idx + i}'),
                'title': doc.get('title', ''),
                'category': doc.get('category', ''),
                'content': doc.get('content', ''),
                'keywords': doc.get('keywords', [])
            })
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Search for similar documents"""
        if self.index is None:
            return []
        
        # Create query embedding
        query_embedding = self.create_embeddings([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.metadata)))
        
        # Return results with metadata
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                results.append((self.metadata[idx], float(score)))
        
        return results
    
    def save_index(self, save_path: Optional[str] = None):
        """Save the FAISS index and metadata"""
        if save_path is None:
            save_path = self.index_path
        
        # Save FAISS index
        faiss.write_index(self.index, save_path)
        
        # Save metadata
        metadata_path = save_path.replace('.faiss', '_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def load_index(self, load_path: Optional[str] = None):
        """Load the FAISS index and metadata"""
        if load_path is None:
            load_path = self.index_path
        
        if not os.path.exists(load_path):
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(load_path)
            
            # Load metadata
            metadata_path = load_path.replace('.faiss', '_metadata.pkl')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
            
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict]:
        """Get document by ID"""
        for metadata in self.metadata:
            if metadata['doc_id'] == doc_id:
                return metadata
        return None
    
    def get_documents_by_category(self, category: str) -> List[Dict]:
        """Get all documents for a specific category"""
        return [doc for doc in self.metadata if doc['category'] == category]
    
    def update_document(self, doc_id: str, new_document: Dict):
        """Update an existing document"""
        for i, metadata in enumerate(self.metadata):
            if metadata['doc_id'] == doc_id:
                # Update metadata
                self.metadata[i].update(new_document)
                
                # Rebuild index (simplified approach)
                self._rebuild_index()
                break
    
    def _rebuild_index(self):
        """Rebuild the entire index from current documents"""
        if not self.documents:
            return
        
        # Create new index
        embedding_dim = len(self.create_embeddings(["test"])[0])
        self.index = faiss.IndexFlatIP(embedding_dim)
        
        # Re-add all documents
        texts = []
        for doc in self.documents:
            text = f"{doc['title']} {doc['content']} {' '.join(doc['keywords'])}"
            texts.append(text)
        
        embeddings = self.create_embeddings(texts)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        return {
            'total_documents': len(self.metadata),
            'categories': list(set(doc['category'] for doc in self.metadata)),
            'index_size': self.index.ntotal if self.index else 0,
            'embedding_model': self.model_name
        }
