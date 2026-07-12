# local-doc-chat

A fully offline, privacy-preserving Retrieval-Augmented Generation (RAG) chatbot that allows you to upload document files (`.pdf`, `.docx`, `.txt`) and ask questions about them locally. No external APIs or active internet connections are used at runtime.

---

## 🔒 Privacy as a Feature
Because this system runs completely offline, **no data ever leaves your machine**. All documents are parsed, split, embedded, indexed, and processed by the LLM entirely on your local hardware. This makes it suitable for analyzing sensitive, proprietary, or private documents that cannot be uploaded to third-party cloud LLM providers.

---

## 🏗️ Architecture Flow
The RAG pipeline operates as follows:
```
[ Upload Document (.pdf/.docx/.txt) ]
               │
               ▼
[ Text Extraction (pypdf / python-docx) ]
               │
               ▼
[ Sliding Window Chunking (~500 chars, 50 overlap) ]
               │
               ▼
[ Local Embeddings Generation (sentence-transformers: all-MiniLM-L6-v2) ]
               │
               ▼
[ Local Vector Indexing (FAISS L2 Flat Index) ]
               │
   ┌───────────┴───────────┐
   ▼                       ▼
[ User Query ] ──► [ FAISS Semantic Search (Top K Chunks) ]
                           │
                           ▼
             [ Strict Local System Prompt Formulation ]
                           │
                           ▼
             [ Local Inference (Ollama: llama3.2:3b) ]
                           │
                           ▼
             [ Answer Generation + Source Citation ]
```

---

## ⚙️ Setup & Installation

### 1. Prerequisites
- **Ollama**: Download and install Ollama for your operating system: [https://ollama.com/download](https://ollama.com/download)
- **Python**: Version 3.9 or higher.

### 2. Pull the LLM Model
Start the Ollama application or service, then run the following in your terminal to download the default model:
```bash
ollama pull llama3.2:3b
```

### 3. Install Python Dependencies
Install the required packages using the offline-friendly, no-cache flag:
```bash
pip install --no-cache-dir -r requirements.txt
```

### 4. Cache the Embeddings Model
Run the test script once while connected to the internet to download and cache the local sentence-transformer weights (`all-MiniLM-L6-v2`):
```bash
python3 test_embed_store.py
```
After this initial download, the model will run completely offline.

---

## 🚀 Running the Application

1. Make sure your local Ollama service is active.
2. Launch the Streamlit server:
   ```bash
   streamlit run app/main.py --server.port 8502
   ```
3. Open your browser and navigate to `http://localhost:8502`.

---

## ⚠️ Limitations
- **Model Size vs. Quality**: Uses a lightweight 3-billion parameter model (`llama3.2:3b`) to ensure smooth local inference on consumer hardware. Response quality and reasoning capability may be lower than larger, cloud-hosted models (e.g., GPT-4).
- **Scope**: Designed for single-document analysis per session.
- **Persistence**: FAISS indices and document chunks do not persist across multiple user sessions; reloading the application resets the workspace state.
