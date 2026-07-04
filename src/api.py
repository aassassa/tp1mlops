import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from src.ingestion import DocumentIngestionPipeline
from src.retrieval import RAGPipeline

app = FastAPI(
    title="Intelligent RAG MLOps Platform API",
    description="API FastAPI pour l'exploitation intelligente des documents d'entreprise avec RAG et MLOps",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration des répertoires de stockage temporaire
UPLOAD_DIR = "./data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Configuration des variables d'environnement
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "enterprise_documents")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 4
    filters: Dict[str, Any] = None

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]

@app.on_event("startup")
def startup_event():
    # Initialiser le client Qdrant et s'assurer que la collection existe
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        # Vérifier si la collection existe, sinon la créer
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

@app.get("/")
def read_root():
    return {
        "message": "Bienvenue sur l'API de la Plateforme RAG MLOps d'Entreprise",
        "status": "online",
        "docs_url": "/docs"
    }

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

@app.post("/ingest", status_code=202)
async def ingest_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
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
    def process_ingestion(path: str):
        pipeline = DocumentIngestionPipeline(
            qdrant_host=QDRANT_HOST,
            qdrant_port=QDRANT_PORT,
            collection_name=COLLECTION_NAME
        )
        pipeline.run_pipeline(path)

    background_tasks.add_task(process_ingestion, file_path)
    
    return {
        "filename": file.filename,
        "status": "file_uploaded",
        "message": "Le document a été téléversé avec succès. L'ingestion et l'indexation ont démarré."
    }

@app.post("/query", response_model=QueryResponse)
async def query_rag_endpoint(request: QueryRequest):
    pipeline = RAGPipeline(
        qdrant_host=QDRANT_HOST,
        qdrant_port=QDRANT_PORT,
        collection_name=COLLECTION_NAME
    )
    result = pipeline.run(request.query, request.top_k, request.filters)
    return QueryResponse(
        query=result["query"],
        answer=result["answer"],
        sources=[{
            "text": s["text"],
            "metadata": s["metadata"]
        } for s in result["sources"]]
    )

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
