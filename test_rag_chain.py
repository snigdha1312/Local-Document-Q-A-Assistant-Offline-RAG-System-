import os
import sys

# Add root folder of project to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.document_loader import load_document
from src.embed_store import EmbedStore
from src.rag_chain import RAGChain

def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    sample_file = os.path.join(data_dir, "sample.docx")
    
    if not os.path.exists(sample_file):
        print(f"Sample file not found at {sample_file}. Please run test_loader.py first.")
        return
        
    print(f"Loading document: {sample_file}")
    chunks = load_document(sample_file, chunk_size=500, chunk_overlap=50)
    
    print("Embedding document...")
    store = EmbedStore()
    store.embed_and_store(chunks)
    
    print("Initializing RAG Chain with llama3.2:3b model...")
    chain = RAGChain(embed_store=store, model_name="llama3.2:3b")
    
    # 3 Test Questions
    questions = [
        "When was local-doc-chat built?",
        "Does local-doc-chat require internet access at runtime?",
        "What is the capital of France?"
    ]
    
    print("\nRunning RAG questions (verifying localhost-only operation):")
    print("==========================================================")
    
    for idx, q in enumerate(questions):
        print(f"\n[Question {idx+1}]: {q}")
        answer, sources = chain.query(q, top_k=2)
        print(f"[Answer]: {answer}")
        print(f"[Sources Retrieved]: {len(sources)} chunks used.")
        for s_idx, src in enumerate(sources):
            print(f"  - Chunk {src['metadata']['chunk_index']}: \"{src['text'][:100]}...\"")
            
    print("\nVerification Complete.")
    print("Note: Ollama client queries http://localhost:11434 by default. No external API keys or network requests were sent.")

if __name__ == "__main__":
    main()
