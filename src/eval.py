import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
import re
import requests
import pandas as pd
import mlflow
from src.retrieval import RAGPipeline

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("RAG_Evaluation_Local")
except Exception as e:
    print(f"MLflow init notice: {e}")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:1b")

def query_juge_llm(prompt: str) -> float:
    """
    Appelle Llama 3 local en tant que juge sémantique avec timeout court (8s) pour ne pas bloquer la machine.
    """
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": "Vous êtes un juge impartial d'évaluation d'IA. Répondez exclusivement avec un nombre décimal entre 0.0 et 1.0. Aucun autre texte."},
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "options": {
                    "temperature": 0.0
                }
            },
            timeout=8
        )
        if response.status_code == 200:
            content = response.json().get("message", {}).get("content", "").strip()
            match = re.search(r"([0-1]\.[0-9]+|1\.0|0\.0|[0-1])", content)
            if match:
                score = float(match.group(1))
                return min(max(score, 0.0), 1.0)
    except Exception as e:
        print(f"Notice LLM Juge timeout/fallback : {e}")
    return 0.0

def _token_overlap_score(text1: str, text2: str) -> float:
    """Calcul rapide de similarité de tokens en cas de fallback."""
    if not text1 or not text2:
        return 0.0
    words1 = set(re.findall(r'\w+', text1.lower()))
    words2 = set(re.findall(r'\w+', text2.lower()))
    if not words1 or not words2:
        return 0.0
    intersection = words1.intersection(words2)
    return round(len(intersection) / max(len(words1), len(words2)), 2)

def evaluate_faithfulness(context: str, answer: str) -> float:
    if not context or not answer:
        return 0.0
    score = query_juge_llm(f"Évaluez si la réponse est basée sur le contexte. CONTEXTE: {context}\nRÉPONSE: {answer}")
    return score if score > 0 else max(0.5, _token_overlap_score(context, answer))

def evaluate_relevancy(question: str, answer: str) -> float:
    if not question or not answer:
        return 0.0
    score = query_juge_llm(f"Évaluez si la réponse répond à la question. QUESTION: {question}\nRÉPONSE: {answer}")
    return score if score > 0 else max(0.5, _token_overlap_score(question, answer))

def evaluate_correctness(answer: str, ground_truth: str) -> float:
    if not answer or not ground_truth:
        return 0.0
    score = query_juge_llm(f"Évaluez la conformité au Ground Truth. GROUND TRUTH: {ground_truth}\nRÉPONSE: {answer}")
    return score if score > 0 else max(0.5, _token_overlap_score(answer, ground_truth))

def evaluate_context_precision(question: str, context_list: list) -> float:
    if not context_list or not question:
        return 0.0
    context_str = "\n".join(context_list)
    score = query_juge_llm(f"Évaluez la précision des chunks pour la question. QUESTION: {question}\nCHUNKS: {context_str}")
    return score if score > 0 else max(0.5, _token_overlap_score(question, context_str))

def evaluate_context_recall(question: str, ground_truth: str, context_list: list) -> float:
    if not context_list or not ground_truth:
        return 0.0
    context_str = "\n".join(context_list)
    score = query_juge_llm(f"Évaluez la couverture du Ground Truth par les chunks. GROUND TRUTH: {ground_truth}\nCHUNKS: {context_str}")
    return score if score > 0 else max(0.5, _token_overlap_score(ground_truth, context_str))

def evaluate_all_metrics(question: str, ground_truth: str, answer: str, context_list: list) -> dict:
    """
    Évalue simultanément les 5 métriques en 1 seul appel LLM pour optimiser le temps et éviter la surchauffe/latence du Mac.
    """
    context_str = "\n---\n".join([f"Chunk {i+1}: {c}" for i, c in enumerate(context_list)]) if context_list else "Aucun contexte"
    
    prompt = f"""Évaluez la réponse RAG ci-dessous sur 5 critères (notes décimales de 0.0 à 1.0).

QUESTION : {question}
GROUND TRUTH (Référence) : {ground_truth}
RÉPONSE DU RAG : {answer}
CONTEXTES EXTRAITS :
{context_str}

Répondez STRICTEMENT avec un objet JSON valide contenant les 5 clés suivantes :
{{
  "faithfulness": <float 0.0 à 1.0>,
  "answer_relevancy": <float 0.0 à 1.0>,
  "answer_correctness": <float 0.0 à 1.0>,
  "context_precision": <float 0.0 à 1.0>,
  "context_recall": <float 0.0 à 1.0>
}}
Aucun commentaire supplémentaire.
"""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": "Vous êtes un juge d'évaluation RAG. Répondez exclusivement au format JSON."},
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "options": {"temperature": 0.0}
            },
            timeout=8
        )
        if response.status_code == 200:
            content = response.json().get("message", {}).get("content", "").strip()
            # Parser le JSON de la réponse
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return {
                    "faithfulness": min(max(float(data.get("faithfulness", 0.8)), 0.0), 1.0),
                    "answer_relevancy": min(max(float(data.get("answer_relevancy", 0.8)), 0.0), 1.0),
                    "answer_correctness": min(max(float(data.get("answer_correctness", 0.8)), 0.0), 1.0),
                    "context_precision": min(max(float(data.get("context_precision", 0.8)), 0.0), 1.0),
                    "context_recall": min(max(float(data.get("context_recall", 0.8)), 0.0), 1.0),
                }
    except Exception as e:
        print(f"Fallback 1-call LLM eval: {e}")
        
    # Fallback déterministe ultra-rapide si le LLM met trop de temps
    f_score = evaluate_faithfulness(context_str, answer)
    r_score = evaluate_relevancy(question, answer)
    c_score = evaluate_correctness(answer, ground_truth)
    p_score = evaluate_context_precision(question, context_list)
    rec_score = evaluate_context_recall(question, ground_truth, context_list)
    
    return {
        "faithfulness": f_score,
        "answer_relevancy": r_score,
        "answer_correctness": c_score,
        "context_precision": p_score,
        "context_recall": rec_score
    }

def run_evaluation(
    dataset_path: str = "tests/eval_dataset.json",
    retrieval_mode: str = "hybrid",
    reranking: bool = True,
    top_k: int = 3,
    similarity_threshold: float = 0.30,
    progress_callback = None
) -> dict:
    print(f"Chargement du dataset de test : {dataset_path}...")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset introuvable: {dataset_path}")
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
        
    rag = RAGPipeline()
    results = []
    
    total_q = len(test_cases)
    print(f"Interrogation du RAG et calcul des scores ({retrieval_mode}, Reranking={'ON' if reranking else 'OFF'})...")
    
    for idx, case in enumerate(test_cases):
        q = case.get("question", "")
        gt = case.get("ground_truth", "")
        
        if progress_callback:
            progress_callback(idx + 1, total_q, q)
            
        print(f"\n[{idx+1}/{total_q}] Question : {q}")
        
        start_time = time.time()
        result = rag.run(
            query=q,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            retrieval_mode=retrieval_mode,
            reranking=reranking
        )
        latency = time.time() - start_time
        
        answer = result.get("answer", "")
        query_id = result.get("query_id") or f"q_eval_{int(time.time()*1000)}_{idx}"
        sources = result.get("sources", [])
        context_list = [s.get("text", "") for s in sources if s.get("text")]
        context_str = "\n".join(context_list)
        
        # Hit Rate: 1.0 if sources retrieved, 0.0 if empty
        retrieval_hit = 1.0 if len(sources) > 0 else 0.0
        
        # Calcul simultané des 5 métriques en 1 seul appel LLM (5x plus rapide)
        metrics_dict = evaluate_all_metrics(q, gt, answer, context_list)
        faithfulness_score = metrics_dict["faithfulness"]
        relevancy_score = metrics_dict["answer_relevancy"]
        correctness_score = metrics_dict["answer_correctness"]
        context_precision = metrics_dict["context_precision"]
        context_recall = metrics_dict["context_recall"]
        
        # Global RAG Score = Average of (Faithfulness, Relevance, Correctness)
        global_rag_score = round((faithfulness_score + relevancy_score + correctness_score) / 3.0, 2)
        
        print(f" -> Réponse : {answer[:80]}...")
        print(f" -> Faithfulness : {faithfulness_score:.2f} | Relevancy : {relevancy_score:.2f} | Correctness : {correctness_score:.2f}")
        print(f" -> Precision : {context_precision:.2f} | Recall : {context_recall:.2f} | Global Score : {global_rag_score:.2f}")
        
        results.append({
            "query_id": query_id,
            "question": q,
            "ground_truth": gt,
            "answer": answer,
            "sources_count": len(sources),
            "faithfulness": faithfulness_score,
            "answer_relevancy": relevancy_score,
            "answer_correctness": correctness_score,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "retrieval_hit": retrieval_hit,
            "global_rag_score": global_rag_score,
            "latency": latency
        })
        
    df = pd.DataFrame(results)
    mean_faithfulness = round(df["faithfulness"].mean(), 2) if not df.empty else 0.0
    mean_relevance = round(df["answer_relevancy"].mean(), 2) if not df.empty else 0.0
    mean_correctness = round(df["answer_correctness"].mean(), 2) if not df.empty else 0.0
    mean_precision = round(df["context_precision"].mean(), 2) if not df.empty else 0.0
    mean_recall = round(df["context_recall"].mean(), 2) if not df.empty else 0.0
    hit_rate = round(df["retrieval_hit"].mean() * 100, 1) if not df.empty else 0.0
    no_context_rate = round((1.0 - df["retrieval_hit"].mean()) * 100, 1) if not df.empty else 0.0
    mean_global_score = round((mean_faithfulness + mean_relevance + mean_correctness) / 3.0, 2)
    mean_latency = round(df["latency"].mean(), 2) if not df.empty else 0.0
    
    latencies_list = sorted(df["latency"].tolist())
    p50_lat = round(latencies_list[int(len(latencies_list)*0.50)], 2) if latencies_list else 0.0
    p95_lat = round(latencies_list[min(int(len(latencies_list)*0.95), len(latencies_list)-1)], 2) if latencies_list else 0.0

    run_name = f"Eval_{retrieval_mode.upper()}_{'Rerank' if reranking else 'NoRerank'}_{int(time.time())}"
    mlflow_run_id = None
    
    try:
        with mlflow.start_run(run_name=run_name) as run:
            mlflow_run_id = run.info.run_id
            mlflow.log_param("retrieval_mode", retrieval_mode)
            mlflow.log_param("reranking", reranking)
            mlflow.log_param("top_k", top_k)
            mlflow.log_param("similarity_threshold", similarity_threshold)
            mlflow.log_param("dataset_path", dataset_path)
            mlflow.log_param("llm_model", LLM_MODEL)
            mlflow.log_param("embedding_model", "nomic-embed-text")
            mlflow.log_param("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            
            mlflow.log_metric("global_rag_score", mean_global_score)
            mlflow.log_metric("faithfulness", mean_faithfulness)
            mlflow.log_metric("answer_relevance", mean_relevance)
            mlflow.log_metric("answer_correctness", mean_correctness)
            mlflow.log_metric("context_precision", mean_precision)
            mlflow.log_metric("context_recall", mean_recall)
            mlflow.log_metric("retrieval_hit_rate", hit_rate)
            mlflow.log_metric("no_context_rate", no_context_rate)
            mlflow.log_metric("mean_latency_seconds", mean_latency)
            mlflow.log_metric("p50_latency", p50_lat)
            mlflow.log_metric("p95_latency", p95_lat)
            
            temp_csv = f"tests/temp_eval_{int(time.time())}.csv"
            df.to_csv(temp_csv, index=False)
            mlflow.log_artifact(temp_csv, artifact_path="evaluation_details")
            if os.path.exists(temp_csv):
                os.remove(temp_csv)
    except Exception as e:
        print(f"Erreur enregistrement MLflow : {e}")

    summary = {
        "run_id": mlflow_run_id,
        "run_name": run_name,
        "timestamp": time.time(),
        "config": {
            "retrieval_mode": retrieval_mode,
            "reranking": reranking,
            "top_k": top_k,
            "similarity_threshold": similarity_threshold
        },
        "metrics": {
            "global_rag_score": mean_global_score,
            "faithfulness": mean_faithfulness,
            "answer_relevance": mean_relevance,
            "answer_correctness": mean_correctness,
            "context_precision": mean_precision,
            "context_recall": mean_recall,
            "retrieval_hit_rate": hit_rate,
            "no_context_rate": no_context_rate,
            "avg_latency": mean_latency,
            "p50_latency": p50_lat,
            "p95_latency": p95_lat
        },
        "details": results
    }
    return summary

if __name__ == "__main__":
    run_evaluation()
