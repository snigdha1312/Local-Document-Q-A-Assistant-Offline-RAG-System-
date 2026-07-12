import os
import sys

# Add root folder of project to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.document_loader import load_document

def main():
    # Use data/sample.docx if it exists, otherwise look for any file in data/
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        return
        
    files = [f for f in os.listdir(data_dir) if f.endswith(('.pdf', '.docx', '.txt'))]
    if not files:
        print(f"No sample files (.pdf, .docx, .txt) found in {data_dir}")
        print("Please place a sample file in the data/ directory.")
        return
        
    sample_file = os.path.join(data_dir, files[0])
    print(f"Loading sample file: {sample_file}")
    
    try:
        chunks = load_document(sample_file, chunk_size=500, chunk_overlap=50)
        print("\n--- Test Results ---")
        print(f"Total Chunks Created: {len(chunks)}")
        if chunks:
            print("\nFirst Chunk:")
            first_chunk = chunks[0]
            print(f"Metadata: {first_chunk['metadata']}")
            print(f"Content:\n{first_chunk['text']}")
    except Exception as e:
        print(f"Error loading document: {e}")

if __name__ == "__main__":
    main()
