import os
# Définition des variables d'environnement de configuration de Langfuse (doivent être définies avant l'import)
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-909f19cb-aeec-45f2-8e37-0f87f09250a1")
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-9f8c2235-fb7e-4212-a684-2416b40a31e0")
os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

import requests
import re
import json
from typing import List, Dict, Any
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langfuse.decorators import observe
import math

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

class RAGPipeline:
    def __init__(
        self,
        qdrant_host: str = None,
        qdrant_port: int = None,
        collection_name: str = None,
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "llama3.2:1b",
        ollama_base_url: str = None
    ):
        self.collection_name = collection_name or os.getenv("COLLECTION_NAME", "enterprise_documents")
        self.llm_model = llm_model
        self.ollama_base_url = ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        q_host = qdrant_host or os.getenv("QDRANT_HOST", "localhost")
        q_port = qdrant_port or int(os.getenv("QDRANT_PORT", 6333))
        self.client = QdrantClient(host=q_host, port=q_port)
        
        # Initialisation des embeddings via Ollama
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=self.ollama_base_url
        )
        
        # Initialisation du modèle de Reranking (Singleton)
        self.reranker = None
        if CrossEncoder:
            print("[RAGPipeline] Chargement du modèle de Reranking...")
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
            print("[RAGPipeline] Modèle de Reranking chargé.")


    @observe(as_type="span", name="Agent_Routeur")
    def route_query(self, query: str) -> str:
        """
        Agent 1: Analyse la question et la route vers le bon domaine pour filtrer la recherche.
        """
        system_prompt = (
            "Vous êtes un agent routeur IA. Votre rôle est de classifier la question de l'utilisateur dans l'une "
            "des trois catégories suivantes : TELETRAVAIL, SECURITE ou GENERAL. "
            "Répondez avec UNIQUEMENT le mot clé de la catégorie. Aucun autre mot, aucun commentaire."
        )
        user_prompt = (
            f"Question à classifier : {query}\n\n"
            "Catégories :\n"
            "- TELETRAVAIL (si la question concerne le télétravail, les congés, le travail à distance, les compensations financières à distance)\n"
            "- SECURITE (si la question concerne les mots de passe, le phishing, le VPN, les incidents informatiques, la sécurité informatique)\n"
            "- GENERAL (pour tout autre sujet ou en cas de doute)\n\n"
            "Catégorie :"
        )
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/chat",
                json={
                    "model": self.llm_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "stream": False,
                    "options": {"temperature": 0.0}
                },
                timeout=60
            )
            if response.status_code == 200:
                result = response.json().get("message", {}).get("content", "").strip().upper()
                if "TELETRAVAIL" in result:
                    return "TELETRAVAIL"
                elif "SECURITE" in result:
                    return "SECURITE"
            return "GENERAL"
        except Exception as e:
            print(f"[Agent Routeur] Erreur de routage : {e}")
            return "GENERAL"

    def _hybrid_search(self, candidates: List[Dict[str, Any]], query: str, vector_weight: float, keyword_weight: float, mode: str) -> List[Dict[str, Any]]:
        """
        Effectue une recherche locale BM25 sur les candidats denses pré-récupérés (Late Fusion).
        Modulaire : pourra être remplacé par un Sparse Retrieval global (ex: Splade ou Qdrant Sparse) plus tard.
        """
        if not BM25Okapi or mode not in ["hybrid", "keyword"]:
            return candidates
            
        tokenized_corpus = [doc["text"].lower().split() for doc in candidates]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # Normalize Dense Scores
        dense_scores = [doc["score"] for doc in candidates]
        min_dense = min(dense_scores) if dense_scores else 0
        max_dense = max(dense_scores) if dense_scores else 1
        
        # Normalize BM25 Scores
        min_bm25 = min(bm25_scores) if len(bm25_scores) > 0 else 0
        max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 else 1
        
        for i, doc in enumerate(candidates):
            norm_dense = (doc["score"] - min_dense) / (max_dense - min_dense + 1e-6)
            norm_bm25 = (bm25_scores[i] - min_bm25) / (max_bm25 - min_bm25 + 1e-6)
            
            if mode == "hybrid":
                doc["score"] = (norm_dense * vector_weight) + (norm_bm25 * keyword_weight)
            else: # mode == "keyword"
                doc["score"] = norm_bm25
                
        return sorted(candidates, key=lambda x: x["score"], reverse=True)

    @observe(as_type="span", name="Qdrant_Retrieval")
    def retrieve(self, query: str, top_k: int = 4, query_filter: Any = None, query_id: str = None, **kwargs) -> Dict[str, Any]:
        """
        Agent 1: Recherche d'information hybride + Reranking + Final Relevance Check
        """
        import time
        from src.monitoring import monitoring_service
        
        latencies = {"retrieval": 0.0, "hybrid": 0.0, "reranking": 0.0}
        rag_trace = {"query": query, "steps": []}
        
        mode = kwargs.get("mode", "hybrid")
        vector_weight = kwargs.get("vector_weight", 0.70)
        keyword_weight = kwargs.get("keyword_weight", 0.30)
        reranking = kwargs.get("reranking", True)
        rerank_threshold = kwargs.get("rerank_threshold", 0.0)
        similarity_threshold = kwargs.get("similarity_threshold", 0.70)
        top_n_candidates = kwargs.get("top_n_candidates", 20)

        if query_id:
            monitoring_service.record_log("INFO", "RAG", "query_received", query_id)
        
        try:
            # 1. Fetch dense candidates from Qdrant
            t0 = time.perf_counter()
            query_vector = self.embeddings.embed_query(query)
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=top_n_candidates if (mode == "hybrid" or reranking) else top_k
            ).points
            
            candidates = []
            for hit in search_result:
                metadata = hit.payload.get("metadata", {})
                candidates.append({
                    "chunk_id": metadata.get("chunk_id", str(hit.id)),
                    "document_name": metadata.get("source", "Inconnu"),
                    "page": metadata.get("page", None),
                    "section": metadata.get("section", None),
                    "text": hit.payload.get("text", ""),
                    "score": hit.score,
                    "rerank_score": None,
                    "metadata": metadata
                })

            latencies["retrieval"] = round((time.perf_counter() - t0) * 1000, 2)
            rag_trace["steps"].append({"name": "vector_search", "candidates_count": len(candidates), "latency_ms": latencies["retrieval"]})
            
            if query_id:
                monitoring_service.record_log("INFO", "Retrieval", "retrieval_completed", query_id, latencies["retrieval"])

            if not candidates:
                return {"sources": [], "latencies": latencies, "rag_trace": rag_trace}

            # 2. Hybrid Search (Late Fusion with BM25)
            t_hybrid = time.perf_counter()
            candidates = self._hybrid_search(candidates, query, vector_weight, keyword_weight, mode)
            latencies["hybrid"] = round((time.perf_counter() - t_hybrid) * 1000, 2)
            rag_trace["steps"].append({"name": "hybrid_search", "mode": mode, "latency_ms": latencies["hybrid"]})

            if query_id:
                monitoring_service.record_log("INFO", "Hybrid", "hybrid_completed", query_id, latencies["hybrid"])

            # 3. Reranking (Cross-Encoder)
            t_rerank = time.perf_counter()
            if reranking and self.reranker:
                try:
                    pairs = [[query, doc["text"]] for doc in candidates]
                    rerank_scores = self.reranker.predict(pairs)
                    
                    for i, doc in enumerate(candidates):
                        doc["rerank_score"] = float(rerank_scores[i])
                        
                    candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
                except Exception as rerank_e:
                    print(f"Erreur lors du reranking: {rerank_e}")
            latencies["reranking"] = round((time.perf_counter() - t_rerank) * 1000, 2)
            rag_trace["steps"].append({"name": "reranking", "enabled": bool(reranking and self.reranker), "latency_ms": latencies["reranking"]})

            if query_id:
                monitoring_service.record_log("INFO", "Reranking", "reranking_completed", query_id, latencies["reranking"])

            # 4. Final Relevance Check (Distinct thresholds)
            filtered = []
            for c in candidates:
                if reranking and c["rerank_score"] is not None:
                    if c["rerank_score"] >= rerank_threshold:
                        filtered.append(c)
                else:
                    if c["score"] >= similarity_threshold:
                        filtered.append(c)
            
            final_results = filtered[:top_k]
            rag_trace["steps"].append({
                "name": "final_relevance_check",
                "passed_count": len(filtered),
                "sources": [{
                    "chunk_id": c.get("chunk_id"),
                    "document_name": c.get("document_name"),
                    "page": c.get("page"),
                    "score": round(c.get("score", 0.0), 4),
                    "rerank_score": round(c.get("rerank_score", 0.0), 4) if c.get("rerank_score") is not None else None
                } for c in final_results]
            })
            
            return {"sources": final_results, "latencies": latencies, "rag_trace": rag_trace}
            
        except Exception as e:
            print(f"Erreur lors du retrieval : {e}")
            return {"sources": [], "latencies": latencies, "rag_trace": rag_trace}

    @observe(as_type="generation", name="Agent_Synthetiseur")
    def generate(self, query: str, sources: List[Dict[str, Any]], **kwargs) -> str:
        """
        Agent 2: Rédige une réponse brouillon en utilisant le contexte récupéré.
        """
        if not sources:
            context_str = "Aucun contexte trouvé."
        else:
            context_str = "\n\n".join([
                f"[Source: {s['metadata'].get('source', 'Inconnu')}, Page: {s['metadata'].get('page', 'N/A')}]\n{s['text']}"
                for s in sources
            ])
            
        system_prompt = (
            "Tu es un assistant d'entreprise professionnel et clair. "
            "Réponds à la question de manière fluide avec une phrase complète et bien formulée en te basant sur le contexte fourni. "
            "Ne rajoute jamais de phrase de refus si le contexte contient la réponse."
        )
        
        user_prompt = (
            f"CONTEXTE:\n{context_str}\n\n"
            f"QUESTION: {query}\n\n"
            f"RÉPONSE:"
        )
        
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/chat",
                json={
                    "model": self.llm_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "stream": False,
                    "options": {"temperature": 0.0}
                },
                timeout=60
            )
            if response.status_code == 200:
                ans = response.json().get("message", {}).get("content", "Erreur lors de la lecture du résultat LLM.").strip()
                for phrase in [
                    "L'information n'est pas disponible dans les documents fournis.",
                    "L'information n'est pas disponible dans les documents fournis",
                    "L'information n'est pas disponible dans les documents fournis:",
                    "L'information n'est pas disponible."
                ]:
                    if phrase in ans and len(ans) > len(phrase) + 5:
                        ans = ans.replace(phrase, "").strip()
                return ans
            return f"Erreur Ollama API : {response.status_code}"
        except Exception as e:
            return f"Impossible de contacter le LLM : {e}"

    @observe(as_type="span", name="Agent_Evaluateur")
    def evaluate_draft(self, query: str, context: str, draft: str) -> Dict[str, Any]:
        """
        Agent 3: Agent évaluateur (Self-RAG) qui vérifie s'il y a des hallucinations ou du hors-sujet.
        """
        system_prompt = (
            "Vous êtes un agent critique de contrôle qualité. Votre rôle est d'analyser une proposition de réponse "
            "par rapport à une question et à un contexte de référence. "
            "Vous devez soit approuver la réponse, soit la rejeter en fournissant une remarque de correction précise."
        )
        user_prompt = f"""Examinez la réponse suivante :
        
        CONTEXTE DE RÉFÉRENCE :
        {context}
        
        QUESTION DE L'UTILISATEUR :
        {query}
        
        PROPOSITION DE RÉPONSE :
        {draft}
        
        Règles d'évaluation :
        1. Si la réponse est fidèle au contexte (aucune invention/hallucination) et répond à la question, écrivez exactement :
        STATUS: APPROVED
        
        2. Si la réponse est hors-sujet, ou si elle invente des faits absents du contexte, ou si elle dit qu'elle ne sait pas alors que l'information est présente, écrivez :
        STATUS: REJECTED | FEEDBACK: [Expliquez ici précisément ce qui est faux ou manquant et comment corriger]
        """
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/chat",
                json={
                    "model": self.llm_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "stream": False,
                    "options": {"temperature": 0.0}
                },
                timeout=60
            )
            if response.status_code == 200:
                content = response.json().get("message", {}).get("content", "").strip()
                if "STATUS: APPROVED" in content or "APPROVED" in content.upper() and "REJECTED" not in content.upper():
                    return {"status": "APPROVED", "feedback": ""}
                
                # Extraction du feedback
                feedback = "La réponse contient des incohérences ou n'est pas claire."
                match = re.search(r"FEEDBACK:\s*(.*)", content, re.IGNORECASE)
                if match:
                    feedback = match.group(1).strip()
                return {"status": "REJECTED", "feedback": feedback}
            return {"status": "APPROVED", "feedback": ""}
        except Exception as e:
            print(f"[Agent Évaluateur] Erreur de validation : {e}")
            return {"status": "APPROVED", "feedback": ""}

    @observe(as_type="generation", name="Agent_Correcteur")
    def generate_corrected(self, query: str, context: str, draft: str, feedback: str) -> str:
        """
        Agent 2 (Correction) : Réécrit la réponse en fonction des remarques du critique.
        """
        system_prompt = (
            "Vous êtes un assistant IA chargé de réécrire une réponse. Vous devez corriger la réponse "
            "précédente en suivant à la lettre les remarques de l'agent de contrôle qualité."
        )
        user_prompt = f"""Voici les détails de la correction :
        
        CONTEXTE DE RÉFÉRENCE :
        {context}
        
        QUESTION POSÉE :
        {query}
        
        BROUILLON REJETÉ :
        {draft}
        
        REMARQUE DE L'ÉVALUATEUR :
        {feedback}
        
        Consigne : Réécrivez la réponse en corrigeant l'erreur signalée. Ne gardez que des informations vérifiables dans le contexte.
        RÉPONSE CORRIGÉE :"""
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/chat",
                json={
                    "model": self.llm_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "stream": False,
                    "options": {"temperature": 0.0}
                },
                timeout=60
            )
            if response.status_code == 200:
                return response.json().get("message", {}).get("content", "")
            return draft
        except Exception as e:
            print(f"[Agent Correcteur] Erreur de correction : {e}")
            return draft

    @observe(name="RAG_MultiAgent_Query")
    def run(self, query: str, top_k: int = 4, filters: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """
        Orchestration du Système Multi-Agent avec tracing sémantique décoré par Langfuse.
        """
        # 1. Agent Routeur
        domain = self.route_query(query)
        print(f"[SMA] Agent Routeur : Catégorie détectée -> {domain}")
        
        # Traduction du domaine en filtre Qdrant (recherche globale sur l'ensemble des documents indexés)
        qdrant_filter = None
            
        # 2. Agent de Recherche
        retrieval_result = self.retrieve(
            query=query, 
            top_k=top_k, 
            query_filter=qdrant_filter,
            query_id=kwargs.get("query_id"),
            similarity_threshold=kwargs.get("similarity_threshold", 0.70),
            mode=kwargs.get("retrieval_mode", "hybrid"),
            vector_weight=kwargs.get("vector_weight", 0.70),
            keyword_weight=kwargs.get("keyword_weight", 0.30),
            reranking=kwargs.get("reranking", True),
            rerank_threshold=kwargs.get("rerank_threshold", 0.0),
            top_n_candidates=kwargs.get("top_n_candidates", 20)
        )
        sources = retrieval_result["sources"]
        latencies = retrieval_result["latencies"]
        rag_trace = retrieval_result.get("rag_trace", {})
        
        # Short-circuit si aucun contexte (NO LLM CALL)
        if not sources:
            print("[SMA] Aucun contexte pertinent trouvé. Short-circuit LLM activé.")
            return {
                "query": query,
                "answer": "L'information n'est pas disponible dans les documents fournis.",
                "sources": [],
                "metadata": {
                    "routing_domain": domain,
                    "evaluation_status": "SHORT_CIRCUIT",
                    "critic_feedback": "Aucun document n'a passé le seuil de pertinence.",
                    "latencies": latencies,
                    "rag_trace": rag_trace
                }
            }
        
        # 3. Agent Synthétiseur (Premier brouillon)
        draft = self.generate(query, sources, system_prompt=kwargs.get("system_prompt"))
        
        # 4. Agent Évaluateur (Self-RAG)
        context_str = "\n\n".join([s["text"] for s in sources]) if sources else "Aucun contexte."
        eval_result = self.evaluate_draft(query, context_str, draft)
        print(f"[SMA] Agent Évaluateur : Statut -> {eval_result['status']}")
        
        final_answer = draft
        eval_status = "APPROVED_FIRST_TRY"
        feedback = ""
        
        # Si rejeté, boucle de correction unique (1 essai)
        if eval_result["status"] == "REJECTED":
            feedback = eval_result["feedback"]
            print(f"[SMA] Correction en cours avec remarque : {feedback}")
            final_answer = self.generate_corrected(query, context_str, draft, feedback)
            eval_status = "CORRECTED"
            
            # Validation de la correction (log local uniquement)
            final_eval = self.evaluate_draft(query, context_str, final_answer)
            print(f"[SMA] Validation finale de la correction -> {final_eval['status']}")
            

            
        return {
            "query": query,
            "answer": final_answer,
            "sources": sources,
            "metadata": {
                "routing_domain": domain,
                "evaluation_status": eval_status,
                "critic_feedback": feedback,
                "latencies": latencies,
                "rag_trace": rag_trace
            }
        }

    @observe(name="RAG_MultiAgent_Stream")
    def run_stream(self, query: str, top_k: int = 4, filters: Dict[str, Any] = None, **kwargs):
        """
        Orchestration du RAG en mode streaming.
        """
        # 1. Agent Routeur
        domain = self.route_query(query)
        print(f"[SMA Stream] Agent Routeur : Catégorie -> {domain}")
        
        # Traduction du domaine en filtre Qdrant
        qdrant_filter = None
        if domain == "TELETRAVAIL":
            qdrant_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.source",
                        match=models.MatchValue(value="politique_rh_et_teletravail_owlmedia.txt")
                    )
                ]
            )
        elif domain == "SECURITE":
            qdrant_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.source",
                        match=models.MatchValue(value="charte_securite_informatique_owlmedia.txt")
                    )
                ]
            )
            
        # 2. Agent de Recherche
        retrieval_result = self.retrieve(
            query=query, 
            top_k=top_k, 
            query_filter=qdrant_filter,
            query_id=kwargs.get("query_id"),
            similarity_threshold=kwargs.get("similarity_threshold", 0.70),
            mode=kwargs.get("retrieval_mode", "hybrid"),
            vector_weight=kwargs.get("vector_weight", 0.70),
            keyword_weight=kwargs.get("keyword_weight", 0.30),
            reranking=kwargs.get("reranking", True),
            rerank_threshold=kwargs.get("rerank_threshold", 0.0),
            top_n_candidates=kwargs.get("top_n_candidates", 20)
        )
        sources = retrieval_result["sources"]
        latencies = retrieval_result["latencies"]
        rag_trace = retrieval_result.get("rag_trace", {})
        
        # Envoyer d'abord les métadonnées (sources, domaine, latencies, rag_trace)
        yield f"metadata:{json.dumps({'sources': sources, 'metadata': {'routing_domain': domain, 'evaluation_status': 'STREAMING', 'latencies': latencies, 'rag_trace': rag_trace}})}\n"

        # Short-circuit si aucun contexte (NO LLM CALL)
        if not sources:
            print("[SMA Stream] Aucun contexte pertinent trouvé. Short-circuit LLM activé.")
            fallback = "L'information n'est pas disponible dans les documents fournis."
            # On stream le fallback très rapidement
            for word in fallback.split():
                yield f"token:{word} \n"
                import time
                time.sleep(0.05)
            # Fin du stream
            yield f"latency_llm:0\n"
            return
            
        # 3. Agent Synthétiseur (Streaming depuis Ollama)
        context_str = "\n\n".join([
            f"[Source: {s['metadata'].get('source', 'Inconnu')}, Page: {s['metadata'].get('page', 'N/A')}]\n{s['text']}"
            for s in sources
        ])
            
        system_prompt = (
            "Tu es un assistant d'entreprise professionnel et clair. "
            "Réponds à la question de manière fluide avec une phrase complète et bien formulée en te basant sur le contexte fourni. "
            "Ne rajoute jamais de phrase de refus si le contexte contient la réponse."
        )
        
        user_prompt = (
            f"CONTEXTE:\n{context_str}\n\n"
            f"QUESTION: {query}\n\n"
            f"RÉPONSE:"
        )
        
        import time
        t_llm = time.perf_counter()
        
        try:
            self._stream_synthesize(system_prompt, user_prompt, response_generator=None)
            response = requests.post(
                f"{self.ollama_base_url}/api/chat",
                json={
                    "model": self.llm_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "stream": True,
                    "options": {"temperature": 0.0}
                },
                stream=True,
                timeout=60
            )
            
            if response.status_code == 200:
                full_tokens = []
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line.decode("utf-8"))
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            full_tokens.append(content)
                            yield f"token:{content}\n"
                        if chunk.get("done", False):
                            latencies["llm"] = round((time.perf_counter() - t_llm) * 1000, 2)
                            yield f"latency_llm:{latencies['llm']}\n"
            else:
                yield f"error:Erreur Ollama API : {response.status_code}\n"
        except Exception as e:
            yield f"error:Impossible de contacter le LLM : {e}\n"

    @observe(as_type="generation", name="Agent_Synthetiseur")
    def _stream_synthesize(self, system_prompt: str, user_prompt: str, response_generator=None):
        return "Streaming generation tracked in Langfuse"

if __name__ == "__main__":
    # Test local rapide
    rag = RAGPipeline()
    print(rag.run("Les stagiaires ont-ils droit au télétravail chez TechInnov ?"))
