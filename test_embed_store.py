import os
import sys

# Add root folder of project to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.document_loader import load_document
from src.embed_store import EmbedStore

def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    files = [f for f in os.listdir(data_dir) if f.endswith(('.pdf', '.docx', '.txt'))]
    if not files:
        print("No sample files found in data/. Please run test_loader.py first to create sample.docx.")
        return
        
    sample_file = os.path.join(data_dir, files[0])
    print(f"Loading sample file: {sample_file}")
    
    # 1. Load document chunks
    chunks = load_document(sample_file, chunk_size=500, chunk_overlap=50)
    print(f"Loaded {len(chunks)} chunks.")
    
    # 2. Embed and Store chunks
    store = EmbedStore()
    print("Embedding chunks and building FAISS index...")
    store.embed_and_store(chunks)
    
    # 3. Test saving the index
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db")
    print(f"Saving index to: {save_path}")
    store.save_index(save_path)
    
    # 4. Test loading the index in a fresh store
    print("Loading index into a fresh EmbedStore instance...")
    new_store = EmbedStore()
    success = new_store.load_index(save_path)
    if not success:
        print("Error: Failed to load index.")
        return
    print("Index loaded successfully!")
    
    # 5. Run a test query
    query = "offline"
    print(f"Running test query: '{query}'")
    results = new_store.search(query, top_k=2)
    
    print("\n--- Query Results ---")
    for idx, (chunk, score) in enumerate(results):
        print(f"\nResult {idx+1} (score/distance: {score:.4f}):")
        print(f"Metadata: {chunk['metadata']}")
        print(f"Text:\n{chunk['text']}")
        
if __name__ == "__main__":
    main()
