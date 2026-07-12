import sys
import os
import streamlit as st

# Ensure the root directory is in the path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_loader import load_document
from src.embed_store import EmbedStore
from src.rag_chain import RAGChain

# Set up page configurations
st.set_page_config(
    page_title="local-doc-chat | Offline RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    /* Main Background and Colors */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1E293B !important;
        border-right: 1px solid #334155;
    }
    
    /* Input box custom border */
    .stTextInput input {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
        border: 1px solid #475569 !important;
        border-radius: 8px !important;
    }
    .stTextInput input:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 1px #3B82F6 !important;
    }
    
    /* Buttons Customization */
    .stButton>button {
        background-color: #3B82F6 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2563EB !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    /* Card design for metadata and status */
    .status-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    
    .source-card {
        background-color: #1E293B;
        border-left: 4px solid #3B82F6;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.75rem;
        font-size: 0.9rem;
    }
    
    .source-title {
        font-weight: bold;
        color: #60A5FA;
        margin-bottom: 0.25rem;
    }
    
    /* Header logo and subtitle */
    .header-container {
        text-align: center;
        padding: 2rem 0;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(to right, #60A5FA, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        color: #94A3B8;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- SYSTEM STARTUP CHECKS ---
ollama_ok = False
ollama_error_msg = ""
model_found = False
try:
    import ollama
    local_models = ollama.list()
    ollama_ok = True
    
    if hasattr(local_models, 'models'):
        models_list = local_models.models
    elif isinstance(local_models, dict) and 'models' in local_models:
        models_list = local_models['models']
    else:
        models_list = local_models
    
    for m in models_list:
        name = m.get('name', '') if isinstance(m, dict) else getattr(m, 'model', '')
        if "llama3.2:3b" in name or name.startswith("llama3.2:3b"):
            model_found = True
            break
except Exception as e:
    ollama_error_msg = str(e)

embeddings_cached = False
embeddings_error_msg = ""
try:
    from sentence_transformers import SentenceTransformer
    # Verify embedding model is cached locally without downloading
    SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
    embeddings_cached = True
except Exception as e:
    embeddings_error_msg = str(e)

# Initialize Session States
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_filename" not in st.session_state:
    st.session_state.processed_filename = None
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = RAGChain(model_name="llama3.2:3b")

rag_chain = st.session_state.rag_chain

# Sidebar Interface
with st.sidebar:
    st.markdown("<div class='header-title' style='font-size: 1.8rem;'>local-doc-chat</div>", unsafe_allow_html=True)
    st.info("🔒 Running fully offline — Ollama + local embeddings, no data leaves this machine")
    st.markdown("---")
    
    # 1. System Status Indicator
    st.markdown("### System Status")
    
    if ollama_ok:
        if model_found:
            st.markdown(
                """
                <div style='display: flex; align-items: center; gap: 8px; color: #10B981; font-weight: 600;'>
                    <span style='height: 12px; width: 12px; background-color: #10B981; border-radius: 50%; display: inline-block;'></span>
                    Ollama: Online (llama3.2:3b)
                </div>
                """, 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div style='display: flex; align-items: center; gap: 8px; color: #F59E0B; font-weight: 600;'>
                    <span style='height: 12px; width: 12px; background-color: #F59E0B; border-radius: 50%; display: inline-block;'></span>
                    Ollama: Online (llama3.2:3b missing)
                </div>
                """, 
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            """
            <div style='display: flex; align-items: center; gap: 8px; color: #EF4444; font-weight: 600;'>
                <span style='height: 12px; width: 12px; background-color: #EF4444; border-radius: 50%; display: inline-block;'></span>
                Ollama: Offline
            </div>
            """, 
            unsafe_allow_html=True
        )

    if embeddings_cached:
        st.markdown(
            """
            <div style='display: flex; align-items: center; gap: 8px; color: #10B981; font-weight: 600; margin-top: 5px;'>
                <span style='height: 12px; width: 12px; background-color: #10B981; border-radius: 50%; display: inline-block;'></span>
                Embeddings: Cached (all-MiniLM-L6-v2)
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style='display: flex; align-items: center; gap: 8px; color: #EF4444; font-weight: 600; margin-top: 5px;'>
                <span style='height: 12px; width: 12px; background-color: #EF4444; border-radius: 50%; display: inline-block;'></span>
                Embeddings: Missing cache
            </div>
            """, 
            unsafe_allow_html=True
        )

    st.markdown("---")
    
    # 2. File Upload Area
    st.markdown("### Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF, Word, or Text document", 
        type=["pdf", "docx", "txt"],
        help="The document is processed and embedded entirely on your device."
    )
    
    # Advanced Settings
    with st.expander("Advanced RAG Settings"):
        chunk_size = st.slider("Chunk Size (characters)", min_value=300, max_value=2000, value=1000, step=100)
        chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=50, max_value=500, value=200, step=50)
        num_retrieve = st.slider("Top Chunks to Retrieve", min_value=2, max_value=8, value=4)
        
    st.markdown("---")
    
    # 3. Actions
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Document Loading & Indexing Logic
if uploaded_file is not None:
    # Check if this file has already been processed with the same settings
    file_key = f"{uploaded_file.name}_{chunk_size}_{chunk_overlap}"
    
    if st.session_state.processed_filename != file_key:
        with st.spinner(f"Extracting and indexing '{uploaded_file.name}' locally..."):
            try:
                # Ensure data directory exists
                data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
                os.makedirs(data_dir, exist_ok=True)
                
                # Save uploaded file temporarily to data/
                temp_file_path = os.path.join(data_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process file into chunks
                chunks = load_document(temp_file_path, chunk_size, chunk_overlap)
                
                # Create Vector Store
                vector_store = EmbedStore()
                vector_store.embed_and_store(chunks)
                
                # Save to session state
                st.session_state.vector_store = vector_store
                st.session_state.processed_filename = file_key
                st.session_state.total_chunks = len(chunks)
                
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    
                st.success("Document successfully processed and indexed!")
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
else:
    # If file was removed, clear index
    st.session_state.vector_store = None
    st.session_state.processed_filename = None
    st.session_state.total_chunks = 0

# Main Page Interface
# Header
st.markdown(
    """
    <div class='header-container'>
        <div class='header-title'>Local Document RAG Chatbot</div>
        <div class='header-subtitle'>Chat securely and 100% offline with your files</div>
    </div>
    """, 
    unsafe_allow_html=True
)

# Startup Blocking Notifications
if not ollama_ok:
    st.error("❌ **Startup Verification Failed: Ollama Service Offline**")
    st.markdown(f"""
    This app is designed to run **100% offline**, but requires the local Ollama LLM service to be running.
    
    ### How to start the service:
    1. Open your terminal.
    2. Start the Ollama background service by running:
       ```bash
       ollama serve
       ```
    3. Alternatively, launch the official Ollama desktop application from your Applications folder.
    4. Verify that the required model is pulled by running:
       ```bash
       ollama pull llama3.2:3b
       ```
    
    *Connection error details: `{ollama_error_msg}`*
    """)
    st.stop()

if not embeddings_cached:
    st.error("❌ **Startup Verification Failed: Local Embedding Model Missing**")
    st.markdown(f"""
    The embedding model `all-MiniLM-L6-v2` was not found in your local Hugging Face cache.
    
    To run 100% offline, this model must be downloaded once during setup:
    1. Connect your machine to the internet.
    2. Run the test script in your terminal to download and cache the model:
       ```bash
       python3 test_embed_store.py
       ```
    3. Once cached, you can disconnect from the internet and run the application fully offline.
    
    *Cache check details: `{embeddings_error_msg}`*
    """)
    st.stop()

# App Content Layout
if st.session_state.vector_store is None:
    # Welcome / Empty State
    st.info("👈 Please upload a PDF, DOCX, or TXT file in the sidebar to start chatting.")
    
    st.markdown("""
    ### How it works:
    1. **Upload**: Drop in your PDF, Word, or Text document in the sidebar.
    2. **Local Embedding**: The document is split into chunks and converted into vector embeddings using `sentence-transformers/all-MiniLM-L6-v2` locally.
    3. **Local DB**: Chunks are indexed inside a local FAISS database.
    4. **Offline RAG**: Your queries fetch relevant chunks from FAISS and send them alongside the question to your local Ollama LLM (`llama3.2:3b`).
    """)
    
    # Display system check warning if model is missing
    if not model_found:
        st.warning(
            "⚠️ **Warning**: The model `llama3.2:3b` is not available in Ollama. "
            "Please start Ollama and run `ollama pull llama3.2:3b` in your terminal to enable generating answers."
        )
else:
    # Document info card
    st.markdown(
        f"""
        <div class='status-card'>
            <strong>Active Document:</strong> {uploaded_file.name} | 
            <strong>Total Chunks:</strong> {st.session_state.total_chunks} | 
            <strong>Embedding Model:</strong> all-MiniLM-L6-v2 (Local)
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Display Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Display source citations if available
            if "sources" in message and message["sources"]:
                with st.expander("Retrieved Context Sources"):
                    for idx, src in enumerate(message["sources"]):
                        st.markdown(
                            f"""
                            <div class='source-card'>
                                <div class='source-title'>Source Chunk {idx+1} (Page/Char Range: {src['metadata'].get('start_char', 0)}-{src['metadata'].get('end_char', 0)})</div>
                                {src['text']}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
    # Chat Input Box
    if user_query := st.chat_input("Ask a question about your document..."):
        # Display user message in chat
        with st.chat_message("user"):
            st.markdown(user_query)
            
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # 1. Similarity Search
        vector_store = st.session_state.vector_store
        rag_chain.embed_store = vector_store
        retrieved_results = vector_store.search(user_query, k=num_retrieve)
        
        # Format chunks list
        context_chunks = [res[0] for res in retrieved_results]
        
        # 2. Query LLM and Stream Response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            # Stream the query
            response_generator = rag_chain.query_stream(
                query_text=user_query,
                context_chunks=context_chunks
            )
            
            for response_chunk in response_generator:
                full_response += response_chunk
                response_placeholder.markdown(full_response + "▌")
                
            response_placeholder.markdown(full_response)
            
            # Display sources in expander inside the current chat block
            if context_chunks:
                with st.expander("Retrieved Context Sources"):
                    for idx, src in enumerate(context_chunks):
                        st.markdown(
                            f"""
                            <div class='source-card'>
                                <div class='source-title'>Source Chunk {idx+1} (Page/Char Range: {src['metadata'].get('start_char', 0)}-{src['metadata'].get('end_char', 0)})</div>
                                {src['text']}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
        # Save assistant message to chat history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": full_response,
            "sources": context_chunks
        })
