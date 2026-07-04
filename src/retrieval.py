import os
import requests
from typing import List, Dict, Any
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient

class RAGPipeline:
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "enterprise_documents",
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "llama3:latest",
        ollama_base_url: str = "http://localhost:11434"
    ):
        self.collection_name = collection_name
        self.llm_model = llm_model
        self.ollama_base_url = ollama_base_url
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        
        # Initialisation des embeddings via Ollama
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=ollama_base_url
        )

    def retrieve(self, query: str, top_k: int = 4, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Recherche les fragments de documents les plus pertinents dans Qdrant.
        """
        try:
            # 1. Générer l'embedding pour la requête
            query_vector = self.embeddings.embed_query(query)
            
            # 2. Effectuer la recherche dans Qdrant
            # TODO: Implémenter le filtrage sémantique si `filters` est fourni
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k
            )
            
            # 3. Formater les résultats
            sources = []
            for hit in search_result:
                sources.append({
                    "text": hit.payload.get("text", ""),
                    "metadata": hit.payload.get("metadata", {}),
                    "score": hit.score
                })
            return sources
        except Exception as e:
            print(f"Erreur lors du retrieval : {e}")
            return []

    def generate(self, query: str, sources: List[Dict[str, Any]]) -> str:
        """
        Génère une réponse structurée en utilisant le contexte récupéré et le LLM local (Ollama).
        """
        if not sources:
            context_str = "Aucun contexte trouvé."
        else:
            context_str = "\n\n".join([
                f"[Source: {s['metadata'].get('source', 'Inconnu')}, Page: {s['metadata'].get('page', 'N/A')}]\n{s['text']}"
                for s in sources
            ])
            
        # Définition du prompt système et utilisateur
        system_prompt = (
            "Vous êtes un assistant IA intelligent et expert de l'entreprise. Votre but est d'aider les employés à "
            "comprendre les documents internes en vous basant UNIQUEMENT sur le contexte fourni ci-dessous. "
            "Si la réponse ne peut pas être trouvée dans le contexte, dites clairement que vous ne savez pas. "
            "Ne faites aucune spéculation et n'inventez rien."
        )
        
        user_prompt = (
            f"CONTEXTE:\n{context_str}\n\n"
            f"QUESTION: {query}\n\n"
            f"RÉPONSE:"
        )
        
        try:
            # Appel direct à l'API d'Ollama
            response = requests.post(
                f"{self.ollama_base_url}/api/chat",
                json={
                    "model": self.llm_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.0  # Température à 0 pour éviter les hallucinations
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("message", {}).get("content", "Erreur lors de la lecture du résultat LLM.")
            else:
                return f"Erreur Ollama API : {response.status_code} - {response.text}"
        except Exception as e:
            return f"Impossible de contacter le LLM Ollama : {e}"

    def run(self, query: str, top_k: int = 4, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Exécute le pipeline RAG complet : Recherche + Génération.
        """
        sources = self.retrieve(query, top_k, filters)
        answer = self.generate(query, sources)
        
        return {
            "query": query,
            "answer": answer,
            "sources": sources
        }

if __name__ == "__main__":
    # Test local rapide
    # rag = RAGPipeline()
    # print(rag.run("Quelle est la politique de télétravail ?"))
    pass
