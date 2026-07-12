import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer

class EmbedStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model. The model is cached locally by sentence-transformers.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = [] # Parallel list mapping FAISS index positions to original chunk text + metadata

    def embed_and_store(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Embed chunks from document_loader.py and store them in a local FAISS index.
        """
        self.chunks = chunks
        if not chunks:
            return

        # Extract texts to embed
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        embeddings = embeddings.astype('float32')
        
        # Get embedding dimension
        dimension = embeddings.shape[1]
        
        # Initialize L2 FAISS index
        self.index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to the index
        self.index.add(embeddings)

    def save_index(self, path: str) -> None:
        """
        Persist the FAISS index and the parallel chunks list to the specified path directory.
        """
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        index_file = os.path.join(path, "index.faiss")
        faiss.write_index(self.index, index_file)
        
        # Save chunks mapping
        chunks_file = os.path.join(path, "chunks.json")
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    def load_index(self, path: str) -> bool:
        """
        Load the persisted FAISS index and the parallel chunks list from the specified path directory.
        """
        index_file = os.path.join(path, "index.faiss")
        chunks_file = os.path.join(path, "chunks.json")
        
        if not (os.path.exists(index_file) and os.path.exists(chunks_file)):
            return False
            
        try:
            # Load FAISS index
            self.index = faiss.read_index(index_file)
            
            # Load chunks mapping
            with open(chunks_file, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
                
            return True
        except Exception as e:
            print(f"Error loading index from {path}: {e}")
            return False

    def search(self, query: str, top_k: int = 4) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search the index for the top_k most relevant chunks.
        Returns a list of tuples containing (chunk, score/distance).
        """
        if self.index is None or not self.chunks:
            return []

        # Embed query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding.astype('float32')

        # Perform search
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            # FAISS returns -1 if there are not enough items in the index
            if idx != -1 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(distances[0][i])))
                
        return results
