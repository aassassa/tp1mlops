import importlib.metadata
importlib.metadata.packages_distributions = lambda: {}

from contextlib import asynccontextmanager
import os
import shutil
import time
import hmac
import hashlib
import json
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from fastapi.responses import StreamingResponse

from src.retrieval import RAGPipeline
from src.monitoring import monitoring_service

SECRET_KEY = "owl-media-secure-key-for-rbac"
security = HTTPBearer()

def generate_token(username: str, role: str) -> str:
    # Jetons expirent après 12 heures
    payload = {
        "username": username,
        "role": role,
        "exp": time.time() + 43200
    }
    payload_b64 = base64.b64encode(json.dumps(payload).encode()).decode()
    signature = hmac.new(SECRET_KEY.encode(), payload_b64.encode(), hashlib.sha256).hexdigest()
    return f"{payload_b64}.{signature}"

def verify_token(token: str) -> dict | None:
    try:
        parts = token.split(".")
        if len(parts) != 2:
            return None
        payload_b64, signature = parts
        expected_signature = hmac.new(SECRET_KEY.encode(), payload_b64.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected_signature):
            return None
        payload = json.loads(base64.b64decode(payload_b64.encode()).decode())
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    payload = verify_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=401,
            detail="Session invalide ou expirée. Veuillez vous reconnecter."
        )
    return payload

def require_role(allowed_roles: List[str]):
    def dependency(user: dict = Depends(get_current_user)):
        if user.get("role") not in allowed_roles:
            raise HTTPException(
                status_code=403,
                detail="Accès refusé : privilèges insuffisants."
            )
        return user
    return dependency

from src.retrieval import RAGPipeline

# Configuration des variables d'environnement
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "enterprise_documents")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

@asynccontextmanager
async def api_lifespan(app: FastAPI):
    # Initialiser le client Qdrant et s'assurer que la collection existe
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]
        
        if COLLECTION_NAME not in collection_names:
            # On utilise par défaut la taille de dimension de nomic-embed-text (768)
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )
            print(f"Collection '{COLLECTION_NAME}' créée avec succès.")
        else:
            print(f"Collection '{COLLECTION_NAME}' déjà existante.")
    except Exception as e:
        print(f"Erreur d'initialisation de Qdrant: {e}. (Assurez-vous que Qdrant tourne sur docker)")
    yield

app = FastAPI(
    title="Intelligent RAG MLOps Platform API",
    description="API FastAPI pour l'exploitation intelligente des documents d'entreprise avec RAG et MLOps",
    version="1.0.0",
    lifespan=api_lifespan
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# Configuration des répertoires de stockage temporaire
UPLOAD_DIR = "./data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

USERS_FILE = "./data/users.json"

INGESTION_STATUS = {}

def load_users_db() -> dict:
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    default_db = {
        "admin": {"password": "admin123", "role": "admin"},
        "user": {"password": "user123", "role": "user"}
    }
    save_users_db(default_db)
    return default_db

def save_users_db(db: dict):
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=4, ensure_ascii=False)

USERS_DB = load_users_db()

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    token: str
    role: str
    username: str

class RegisterRequest(BaseModel):
    username: str
    password: str

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: Dict[str, Any] = None
    similarity_threshold: float = 0.70
    retrieval_mode: str = "hybrid" # vector, keyword, hybrid
    vector_weight: float = 0.70
    keyword_weight: float = 0.30
    reranking: bool = True
    rerank_threshold: float = 0.0
    top_n_candidates: int = 20
    system_prompt: str = None

class QueryResponse(BaseModel):
    query_id: str
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    retrieval: Dict[str, Any]
    latencies: Dict[str, float]
    rag_trace: Dict[str, Any]
    latency: float = 0.0

# Servir l'index du dashboard via StaticFiles à la racine

@app.get("/health")
def health_check():
    # Vérification simple de l'état des services
    qdrant_status = "offline"
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        client.get_collections()
        qdrant_status = "online"
    except Exception:
        pass
        
    return {
        "status": "healthy" if qdrant_status == "online" else "degraded",
        "services": {
            "qdrant": qdrant_status,
            "api": "online"
        }
    }

@app.post("/api/register")
def register_endpoint(request: RegisterRequest):
    username = request.username.strip()
    password = request.password.strip()
    
    if not username or not password:
        raise HTTPException(status_code=400, detail="Nom d'utilisateur et mot de passe requis.")
        
    global USERS_DB
    USERS_DB = load_users_db()
    
    if username in USERS_DB:
        raise HTTPException(status_code=400, detail="Cet utilisateur existe déjà.")
        
    USERS_DB[username] = {
        "password": password,
        "role": "user"
    }
    save_users_db(USERS_DB)
    
    return {"message": "Compte créé avec succès ! Vous pouvez maintenant vous connecter."}

@app.post("/api/login", response_model=LoginResponse)
def login(request: LoginRequest):
    global USERS_DB
    USERS_DB = load_users_db()
    user = USERS_DB.get(request.username)
    if not user or user["password"] != request.password:
        raise HTTPException(status_code=401, detail="Identifiants ou mot de passe incorrect.")
    token = generate_token(request.username, user["role"])
    return {
        "token": token,
        "role": user["role"],
        "username": request.username
    }

@app.post("/ingest", status_code=202)
async def ingest_file(
    file: UploadFile = File(...),
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    use_semantic: bool = False,
    background_tasks: BackgroundTasks = None,
    user: dict = Depends(require_role(["admin"]))
):
    # Validation du type de fichier
    allowed_extensions = [".pdf", ".docx", ".txt"]
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Extension de fichier non supportée. Autorisé : {', '.join(allowed_extensions)}"
        )
        
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Impossible de sauvegarder le fichier : {e}")

    import uuid
    job_id = str(uuid.uuid4())
    filename = file.filename
    INGESTION_STATUS[job_id] = {
        "step": "extraction",
        "progress": 0,
        "details": "Démarrage...",
        "start_time": time.time(),
        "chunks_count": 0,
        "documents_count": 0,
        "filename": filename
    }

    def update_progress(step: str, progress: int, details: str, chunks: int = 0, docs: int = 0):
        if job_id in INGESTION_STATUS:
            INGESTION_STATUS[job_id].update({
                "step": step,
                "progress": progress,
                "details": details
            })
            if chunks:
                INGESTION_STATUS[job_id]["chunks_count"] = chunks
            if docs:
                INGESTION_STATUS[job_id]["documents_count"] = docs

    def process_ingestion(path: str, size: int, overlap: int, semantic: bool, cb):
        from src.ingestion import DocumentIngestionPipeline
        try:
            pipeline = DocumentIngestionPipeline(
                qdrant_host=QDRANT_HOST,
                qdrant_port=QDRANT_PORT,
                collection_name=COLLECTION_NAME,
                ollama_base_url=OLLAMA_BASE_URL
            )
            pipeline.run_pipeline(path, chunk_size=size, chunk_overlap=overlap, use_semantic=semantic, progress_callback=cb)
        except Exception as e:
            cb("error", 0, str(e))

    background_tasks.add_task(process_ingestion, file_path, chunk_size, chunk_overlap, use_semantic, update_progress)
    
    return {
        "job_id": job_id,
        "filename": file.filename,
        "status": "file_uploaded",
        "message": "Le document a été téléversé avec succès. L'ingestion et l'indexation ont démarré."
    }

@app.post("/query", response_model=QueryResponse)
async def query_rag_endpoint(request: QueryRequest, background_tasks: BackgroundTasks, user: dict = Depends(require_role(["user", "admin"]))):
    import time
    import uuid
    start_time = time.perf_counter()
    query_id = f"q_{uuid.uuid4().hex[:8]}"
    
    monitoring_service.record_log("INFO", "API", f"Requête reçue (mode: {request.retrieval_mode})", query_id)
    
    pipeline = RAGPipeline(
        qdrant_host=QDRANT_HOST,
        qdrant_port=QDRANT_PORT,
        collection_name=COLLECTION_NAME,
        ollama_base_url=OLLAMA_BASE_URL
    )
    result = pipeline.run(
        query=request.query, 
        query_id=query_id,
        top_k=request.top_k, 
        filters=request.filters,
        similarity_threshold=request.similarity_threshold,
        retrieval_mode=request.retrieval_mode,
        vector_weight=request.vector_weight,
        keyword_weight=request.keyword_weight,
        reranking=request.reranking,
        rerank_threshold=request.rerank_threshold,
        top_n_candidates=request.top_n_candidates,
        system_prompt=request.system_prompt
    )
    elapsed_time = (time.perf_counter() - start_time) * 1000
    
    # Calculate LLM latency (total - retrieval - hybrid - reranking)
    latencies = result.get("metadata", {}).get("latencies", {})
    llm_lat = elapsed_time - latencies.get("retrieval", 0) - latencies.get("hybrid", 0) - latencies.get("reranking", 0)
    latencies["llm"] = round(llm_lat if llm_lat > 0 else 0, 2)
    latencies["total"] = round(elapsed_time, 2)
    
    rag_trace = result.get("metadata", {}).get("rag_trace", {})
    
    final_cnt = len(result["sources"])
    query_status = "NO_CONTEXT" if final_cnt == 0 else "SUCCESS"

    metrics = {
        "retrieval_mode": request.retrieval_mode,
        "top_k": request.top_k,
        "reranking": request.reranking,
        "latencies": latencies,
        "candidates_count": rag_trace.get("steps", [{}])[0].get("candidates_count", 0) if rag_trace.get("steps") else 0,
        "final_sources_count": final_cnt,
        "status": query_status
    }
    background_tasks.add_task(monitoring_service.record_query, query_id, metrics, rag_trace)
    background_tasks.add_task(monitoring_service.record_log, "SUCCESS", "RAG", f"query_completed ({final_cnt} sources)", query_id, latencies["total"])
    
    return QueryResponse(
        query_id=query_id,
        query=result["query"],
        answer=result["answer"],
        sources=[{
            "text": s.get("text", ""),
            "metadata": s.get("metadata", {}),
            "score": s.get("score", 0.0),
            "rerank_score": s.get("rerank_score", None),
            "chunk_id": s.get("chunk_id", None),
            "document_name": s.get("document_name", None),
            "page": s.get("page", None)
        } for s in result["sources"]],
        retrieval={
            "mode": request.retrieval_mode,
            "top_k": request.top_k,
            "reranking": request.reranking
        },
        latencies=latencies,
        rag_trace=result.get("metadata", {}).get("rag_trace", {}),
        latency=round(elapsed_time / 1000, 3) # for backwards compatibility
    )

@app.post("/query/stream")
async def query_rag_stream_endpoint(request: QueryRequest, background_tasks: BackgroundTasks, user: dict = Depends(require_role(["user", "admin"]))):
    import uuid
    query_id = f"q_{uuid.uuid4().hex[:8]}"
    
    monitoring_service.record_log("INFO", "API", f"query_received (mode: {request.retrieval_mode})", query_id)
    
    pipeline = RAGPipeline(
        qdrant_host=QDRANT_HOST,
        qdrant_port=QDRANT_PORT,
        collection_name=COLLECTION_NAME,
        ollama_base_url=OLLAMA_BASE_URL
    )
    
    def event_generator():
        import time
        start_time = time.perf_counter()
        latencies = {}
        rag_trace = {}
        sources = []
        status = "STREAMING"
        for chunk in pipeline.run_stream(
            query=request.query, 
            query_id=query_id,
            top_k=request.top_k, 
            filters=request.filters,
            similarity_threshold=request.similarity_threshold,
            retrieval_mode=request.retrieval_mode,
            vector_weight=request.vector_weight,
            keyword_weight=request.keyword_weight,
            reranking=request.reranking,
            rerank_threshold=request.rerank_threshold,
            top_n_candidates=request.top_n_candidates,
            system_prompt=request.system_prompt
        ):
            if chunk.startswith("metadata:"):
                try:
                    meta = json.loads(chunk[9:])
                    sources = meta.get("sources", [])
                    status = meta.get("metadata", {}).get("evaluation_status", "SUCCESS")
                    latencies = meta.get("metadata", {}).get("latencies", {})
                    rag_trace = meta.get("metadata", {}).get("rag_trace", {})
                except:
                    pass
            elif chunk.startswith("latency_llm:"):
                try:
                    latencies["llm"] = float(chunk[12:])
                except:
                    pass
            elif chunk.startswith("error:"):
                status = "ERROR"
                
            yield chunk
            
        elapsed_time = (time.perf_counter() - start_time) * 1000
        latencies["total"] = round(elapsed_time, 2)
        
        final_cnt = len(sources)
        if status == "ERROR":
            query_status = "ERROR"
        elif final_cnt == 0 or status == "SHORT_CIRCUIT":
            query_status = "NO_CONTEXT"
        else:
            query_status = "SUCCESS"

        metrics = {
            "retrieval_mode": request.retrieval_mode,
            "top_k": request.top_k,
            "reranking": request.reranking,
            "latencies": latencies,
            "candidates_count": rag_trace.get("steps", [{}])[0].get("candidates_count", 0) if rag_trace.get("steps") else 0,
            "final_sources_count": final_cnt,
            "status": query_status
        }
        monitoring_service.record_query(query_id, metrics, rag_trace)
        
        monitoring_service.record_log("INFO", "LLM", "llm_completed", query_id, latencies.get("llm", 0.0))
        log_level = "ERROR" if query_status == "ERROR" else ("WARNING" if query_status == "NO_CONTEXT" else "SUCCESS")
        monitoring_service.record_log(log_level, "RAG", f"query_completed ({final_cnt} sources)", query_id, latencies["total"])

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/ingestion/status/{job_id}")
def get_ingestion_status(job_id: str, user: dict = Depends(require_role(["admin"]))):
    status = INGESTION_STATUS.get(job_id)
    if not status:
        return {"step": "unknown", "progress": 0, "details": "Statut introuvable"}
    return status

@app.get("/api/documents")
def get_documents_endpoint(user: dict = Depends(require_role(["admin"]))):
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        offset = None
        documents_dict = {}
        
        while True:
            res, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                with_payload=True,
                with_vectors=False,
                offset=offset
            )
            
            for point in res:
                payload = point.payload
                if not payload:
                    continue
                meta = payload.get("metadata", {})
                source = meta.get("source", "Inconnu")
                file_type = meta.get("file_type", source.split(".")[-1] if "." in source else "unknown").upper()
                char_count = len(payload.get("text", ""))
                timestamp = payload.get("timestamp", time.time())
                
                if source not in documents_dict:
                    documents_dict[source] = {
                        "name": source,
                        "type": file_type,
                        "chunks": 0,
                        "size_chars": 0,
                        "indexed_at": timestamp
                    }
                
                documents_dict[source]["chunks"] += 1
                documents_dict[source]["size_chars"] += char_count
                if timestamp > documents_dict[source]["indexed_at"]:
                    documents_dict[source]["indexed_at"] = timestamp
                    
            offset = next_offset
            if not offset or len(res) < 100:
                break
                
        # Format indexed_at as string
        formatted_docs = []
        for doc in documents_dict.values():
            import datetime
            dt = datetime.datetime.fromtimestamp(doc["indexed_at"])
            doc["indexed_at_str"] = dt.strftime("%d/%m/%Y %H:%M")
            formatted_docs.append(doc)
            
        return formatted_docs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def delete_document_points_from_qdrant(source_name: str):
    """
    Deletes all existing points for a given source_name from Qdrant by scrolling IDs to prevent duplicates.
    """
    try:
        from qdrant_client.http import models
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        offset = None
        point_ids = []
        while True:
            res, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                with_payload=True,
                with_vectors=False,
                offset=offset,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.source",
                            match=models.MatchValue(value=source_name)
                        )
                    ]
                )
            )
            for p in res:
                point_ids.append(p.id)
            offset = next_offset
            if not offset or len(res) < 100:
                break
                
        if point_ids:
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=models.PointIdsList(points=point_ids)
            )
            
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.source",
                            match=models.MatchValue(value=source_name)
                        )
                    ]
                )
            )
        )
    except Exception as e:
        print(f"Erreur lors de la suppression des points Qdrant pour {source_name}: {e}")

@app.delete("/api/documents/{source_name}")
def delete_document_endpoint(source_name: str, user: dict = Depends(require_role(["admin"]))):
    try:
        delete_document_points_from_qdrant(source_name)
        return {"status": "success", "message": f"Document '{source_name}' supprimé avec succès."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents/{source_name}/chunks")
def get_document_chunks_endpoint(source_name: str, user: dict = Depends(require_role(["admin"]))):
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        from qdrant_client.http import models
        
        offset = None
        chunks = []
        while True:
            res, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                with_payload=True,
                with_vectors=False,
                offset=offset,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.source",
                            match=models.MatchValue(value=source_name)
                        )
                    ]
                )
            )
            for p in res:
                if p.payload:
                    chunks.append(p.payload)
            offset = next_offset
            if not offset or len(res) < 100:
                break
        
        return chunks
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reindex")
def reindex_document_endpoint(
    background_tasks: BackgroundTasks,
    source_name: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    use_semantic: bool = False,
    user: dict = Depends(require_role(["admin"]))
):
    try:
        # Check if file exists in data/uploads or data/raw or data/
        file_path = os.path.join(UPLOAD_DIR, source_name)
        if not os.path.exists(file_path):
            raw_path = os.path.join("data", "raw", source_name)
            data_path = os.path.join("data", source_name)
            if os.path.exists(raw_path):
                file_path = raw_path
            elif os.path.exists(data_path):
                file_path = data_path
            else:
                raise HTTPException(status_code=404, detail="Le fichier source n'existe plus sur le serveur, re-upload nécessaire.")
            
        import uuid
        job_id = str(uuid.uuid4())
        
        INGESTION_STATUS[job_id] = {
            "step": "extraction",
            "progress": 0,
            "details": "Suppression de l'ancien index...",
            "start_time": time.time(),
            "chunks_count": 0,
            "documents_count": 0,
            "filename": source_name
        }
        
        def process_reindex(path: str, size: int, overlap: int, semantic: bool, cb):
            try:
                # 1. Delete existing vectors first by ID to avoid duplicates
                delete_document_points_from_qdrant(source_name)
                
                # 2. Rerun ingestion pipeline
                from src.ingestion import DocumentIngestionPipeline
                pipeline = DocumentIngestionPipeline(
                    qdrant_host=QDRANT_HOST,
                    qdrant_port=QDRANT_PORT,
                    collection_name=COLLECTION_NAME,
                    ollama_base_url=OLLAMA_BASE_URL
                )
                pipeline.run_pipeline(path, chunk_size=size, chunk_overlap=overlap, use_semantic=semantic, progress_callback=cb)
            except Exception as e:
                cb("error", 0, str(e))
                
        def update_progress(step: str, progress: int, details: str, chunks: int = 0, docs: int = 0):
            if job_id in INGESTION_STATUS:
                INGESTION_STATUS[job_id].update({"step": step, "progress": progress, "details": details})
                if chunks:
                    INGESTION_STATUS[job_id]["chunks_count"] = chunks
                if docs:
                    INGESTION_STATUS[job_id]["documents_count"] = docs
                    
        background_tasks.add_task(process_reindex, file_path, chunk_size, chunk_overlap, use_semantic, update_progress)
        
        return {"status": "success", "job_id": job_id, "message": "Réindexation démarrée."}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
def get_stats_endpoint(user: dict = Depends(require_role(["admin"]))):
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        collection_info = client.get_collection(COLLECTION_NAME)
        points_count = collection_info.points_count
        
        offset = None
        unique_docs = set()
        while True:
            res, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                with_payload=True,
                with_vectors=False,
                offset=offset
            )
            for p in res:
                if p.payload and "metadata" in p.payload:
                    unique_docs.add(p.payload["metadata"].get("source", "Inconnu"))
            offset = next_offset
            if not offset or len(res) < 100:
                break
                
        return {
            "chunks_count": points_count,
            "documents_count": len(unique_docs),
            "qdrant_status": "online",
            "collection_name": COLLECTION_NAME
        }
    except Exception as e:
        return {
            "chunks_count": 0,
            "documents_count": 0,
            "qdrant_status": "offline",
            "error": str(e)
        }

@app.get("/api/health")
def get_health():
    import requests
    services = {}
    
    # 1. Check API
    services["api"] = {"status": "healthy"}

    # 2. Check Qdrant
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2.0)
        client.get_collections()
        services["qdrant"] = {"status": "healthy"}
    except Exception as e:
        services["qdrant"] = {"status": "unhealthy", "details": str(e)}
        
    # 3. Check LLM (Ollama)
    try:
        res = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2.0)
        if res.status_code == 200:
            services["llm"] = {"status": "healthy"}
        else:
            services["llm"] = {"status": "unhealthy", "details": f"Status {res.status_code}"}
    except Exception as e:
        services["llm"] = {"status": "unhealthy", "details": str(e)}

    # 4. Check Embedding (Ollama Embeddings Service)
    try:
        res = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2.0)
        if res.status_code == 200:
            services["embedding"] = {"status": "healthy"}
        else:
            services["embedding"] = {"status": "unhealthy", "details": "Ollama service unavailable"}
    except Exception as e:
        services["embedding"] = {"status": "unhealthy", "details": str(e)}
        
    # 5. Check Reranker
    try:
        from sentence_transformers import CrossEncoder
        services["reranker"] = {"status": "loaded"}
    except Exception as e:
        services["reranker"] = {"status": "unhealthy", "details": str(e)}
    
    overall_status = "healthy" if all(s["status"] in ["healthy", "loaded"] for s in services.values()) else "degraded"
    
    return {
        "status": overall_status,
        "services": services
    }

@app.get("/api/metrics")
def get_metrics(period: str = "24h", user: dict = Depends(require_role(["admin"]))):
    import time
    now = time.time()
    
    if period == "1h":
        start_ts = now - 3600
    elif period == "7d":
        start_ts = now - (7 * 86400)
    elif period == "30d":
        start_ts = now - (30 * 86400)
    else: # 24h
        start_ts = now - 86400
        
    return monitoring_service.get_metrics(start_ts)

@app.get("/api/logs")
def get_logs(level: str = "ALL", limit: int = 50, user: dict = Depends(require_role(["admin"]))):
    return monitoring_service.get_logs(limit=limit, level=level)

@app.get("/api/trace/{query_id}")
def get_trace(query_id: str, user: dict = Depends(require_role(["admin"]))):
    trace = monitoring_service.get_trace(query_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace non trouvée")
    return trace

class EvalRunRequest(BaseModel):
    dataset_path: str = "tests/eval_dataset.json"
    retrieval_mode: str = "hybrid"
    reranking: bool = True
    top_k: int = 3
    similarity_threshold: float = 0.30

EVAL_JOBS: Dict[str, Any] = {}

def process_eval_run_task(job_id: str, request_data: dict):
    from src.eval import run_evaluation
    try:
        def progress_cb(q_idx, q_total, q_text):
            pct = int((q_idx / q_total) * 100)
            if job_id in EVAL_JOBS:
                EVAL_JOBS[job_id].update({
                    "progress": pct,
                    "details": f"Question {q_idx}/{q_total}: {q_text[:40]}...",
                    "completed_questions": q_idx,
                    "total_questions": q_total,
                    "current_question": q_text
                })
            
        res = run_evaluation(
            dataset_path=request_data.get("dataset_path", "tests/eval_dataset.json"),
            retrieval_mode=request_data.get("retrieval_mode", "hybrid"),
            reranking=request_data.get("reranking", True),
            top_k=request_data.get("top_k", 3),
            similarity_threshold=request_data.get("similarity_threshold", 0.30),
            progress_callback=progress_cb
        )
        
        if job_id in EVAL_JOBS:
            EVAL_JOBS[job_id].update({
                "status": "completed",
                "progress": 100,
                "details": "Évaluation terminée avec succès.",
                "end_time": time.time(),
                "results": res
            })
    except Exception as e:
        if job_id in EVAL_JOBS:
            EVAL_JOBS[job_id].update({
                "status": "error",
                "progress": 0,
                "details": f"Erreur d'évaluation: {str(e)}",
                "error": str(e)
            })

@app.post("/api/eval/run")
def start_eval_endpoint(request: EvalRunRequest, background_tasks: BackgroundTasks, user: dict = Depends(require_role(["admin"]))):
    import uuid
    job_id = f"eval_{uuid.uuid4().hex[:8]}"
    
    EVAL_JOBS[job_id] = {
        "job_id": job_id,
        "type": "single",
        "status": "running",
        "progress": 0,
        "details": "Démarrage de l'évaluation...",
        "start_time": time.time(),
        "total_questions": 0,
        "completed_questions": 0,
        "results": None,
        "error": None
    }
    
    background_tasks.add_task(process_eval_run_task, job_id, request.dict())
    return {"job_id": job_id, "status": "running", "message": "Évaluation démarrée."}

@app.get("/api/eval/status/{job_id}")
def get_eval_status_endpoint(job_id: str, user: dict = Depends(require_role(["admin"]))):
    job = EVAL_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Tâche d'évaluation introuvable.")
    
    start = job.get("start_time", time.time())
    end = job.get("end_time", time.time()) if job.get("status") in ["completed", "error"] else time.time()
    job_copy = dict(job)
    job_copy["elapsed_time"] = round(end - start, 1)
    return job_copy

@app.get("/api/eval/history")
def get_eval_history_endpoint(user: dict = Depends(require_role(["admin"]))):
    history = []
    for jid, job in EVAL_JOBS.items():
        history.append({
            "job_id": jid,
            "type": job.get("type"),
            "status": job.get("status"),
            "progress": job.get("progress"),
            "start_time": job.get("start_time"),
            "results_summary": job.get("results", {}).get("metrics") if job.get("results") else None
        })
    return sorted(history, key=lambda x: x.get("start_time", 0), reverse=True)

@app.get("/api/eval/results/{job_id}")
def get_eval_results_endpoint(job_id: str, user: dict = Depends(require_role(["admin"]))):
    job = EVAL_JOBS.get(job_id)
    if not job or not job.get("results"):
        raise HTTPException(status_code=404, detail="Résultats d'évaluation introuvables.")
    return job["results"]

# Servir le dashboard HTML statique (Doit être impérativement monté en DERNIER pour ne pas intercepter les routes /api)
app.mount("/", StaticFiles(directory="src/static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)
