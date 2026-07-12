import ollama
from typing import List, Dict, Any, Tuple, Generator
from src.embed_store import EmbedStore

class RAGChain:
    def __init__(self, embed_store: EmbedStore = None, model_name: str = "llama3.2:3b"):
        """
        Initialize the RAG chain with a reference to the local vector embed store
        and the Ollama model name.
        """
        self.embed_store = embed_store
        self.model_name = model_name

    def is_model_available(self) -> bool:
        """Check if the configured Ollama model is downloaded locally."""
        try:
            local_models = ollama.list()
            if hasattr(local_models, 'models'):
                models_list = local_models.models
            elif isinstance(local_models, dict) and 'models' in local_models:
                models_list = local_models['models']
            else:
                models_list = local_models
            
            for m in models_list:
                name = m.get('name', '') if isinstance(m, dict) else getattr(m, 'model', '')
                if self.model_name in name or name.startswith(self.model_name):
                    return True
            return False
        except Exception as e:
            print(f"Error checking Ollama model status: {e}")
            return False

    def build_system_prompt(self, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Build the RAG system prompt with strict constraints.
        """
        context_str = ""
        for i, chunk in enumerate(context_chunks):
            filename = chunk.get("metadata", {}).get("filename", "Unknown Document")
            context_str += f"\n--- Context Source {i+1} (Document: {filename}) ---\n"
            context_str += chunk["text"] + "\n"

        system_prompt = (
            "You are a local RAG assistant. You must answer the user's question using ONLY the provided Context.\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Answer the question using ONLY the facts present in the Context.\n"
            "2. If the context does not contain the answer or if you are unsure, respond EXACTLY with:\n"
            "   \"The document doesn't contain this information\"\n"
            "3. Do not add outside assumptions, external information, or pleasantries.\n"
            "4. Keep your answer factual, direct, and short.\n\n"
            f"CONTEXT:\n{context_str}"
        )
        return system_prompt

    def query(self, query_text: str, top_k: int = 4) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Takes a user query, retrieves top-k chunks, formats the prompt,
        queries the local Ollama LLM, and returns the generated answer string and source chunks.
        """
        # 1. Retrieve top-k chunks
        results = self.embed_store.search(query_text, top_k=top_k)
        context_chunks = [res[0] for res in results]

        # 2. Build strict prompt
        system_prompt = self.build_system_prompt(context_chunks)

        # 3. Call local Ollama (localhost only)
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query_text}
                ],
                options={"temperature": 0.0} # Lower temperature for factual retrieval
            )
            answer = response.get("message", {}).get("content", "").strip()
            return answer, context_chunks
        except Exception as e:
            error_msg = f"Error communicating with local Ollama: {str(e)}"
            return error_msg, context_chunks

    def query_stream(self, query_text: str, context_chunks: List[Dict[str, Any]]) -> Generator[str, None, None]:
        """
        Helper for streaming responses to Streamlit interfaces.
        Takes pre-retrieved context chunks.
        """
        system_prompt = self.build_system_prompt(context_chunks)
        
        try:
            response_stream = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query_text}
                ],
                options={"temperature": 0.0},
                stream=True
            )
            for chunk in response_stream:
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
        except Exception as e:
            yield f"\nError communicating with local Ollama: {str(e)}"
