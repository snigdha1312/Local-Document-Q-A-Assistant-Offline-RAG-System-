import os
from typing import List, Dict, Any
from pypdf import PdfReader
from docx import Document

def load_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def load_docx(file_path: str) -> str:
    """Extract text from a DOCX file."""
    doc = Document(file_path)
    text = ""
    for para in doc.paragraphs:
        if para.text:
            text += para.text + "\n"
    return text

def load_txt(file_path: str) -> str:
    """Read text from a TXT file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def split_text_sliding_window(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks using a simple sliding window.
    """
    chunks = []
    text_len = len(text)
    
    # If the text is shorter than chunk_size, return it as a single chunk
    if text_len <= chunk_size:
        return [{"text": text.strip(), "start": 0, "end": text_len}]
        
    step = chunk_size - chunk_overlap
    if step <= 0:
        # Fallback to prevent infinite loops if overlap is configured larger than chunk_size
        step = chunk_size // 2

    start = 0
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "start": start,
                "end": end
            })
        if end == text_len:
            break
        start += step
        
    return chunks

def load_document(file_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Accepts PDF, DOCX, and TXT files, extracts text, chunks it, and returns
    chunk objects with metadata (filename, chunk index).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.pdf':
        text = load_pdf(file_path)
    elif ext == '.docx':
        text = load_docx(file_path)
    elif ext == '.txt':
        text = load_txt(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Supported formats are .pdf, .docx, .txt")
        
    if not text.strip():
        raise ValueError("The document is empty or no text could be extracted.")
        
    raw_chunks = split_text_sliding_window(text, chunk_size, chunk_overlap)
    
    filename = os.path.basename(file_path)
    processed_chunks = []
    
    for idx, chunk in enumerate(raw_chunks):
        processed_chunks.append({
            "text": chunk["text"],
            "metadata": {
                "filename": filename,
                "chunk_index": idx,
                "start_char": chunk["start"],
                "end_char": chunk["end"]
            }
        })
        
    return processed_chunks
