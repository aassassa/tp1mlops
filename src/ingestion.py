import os
import time
from typing import List, Dict, Any
import fitz  # PyMuPDF
from docx import Document as DocxDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models

class DocumentIngestionPipeline:
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "enterprise_documents",
        embedding_model: str = "nomic-embed-text",
        ollama_base_url: str = "http://localhost:11434"
    ):
        self.collection_name = collection_name
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        
        # Initialisation des embeddings via Ollama
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=ollama_base_url
        )

    def extract_text(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extrait le texte d'un document (PDF, DOCX, TXT) avec ses métadonnées de base.
        """
        ext = os.path.splitext(file_path)[1].lower()
        documents = []
        filename = os.path.basename(file_path)
        
        try:
            if ext == ".pdf":
                doc = fitz.open(file_path)
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    if text.strip():
                        documents.append({
                            "text": text,
                            "metadata": {
                                "source": filename,
                                "file_path": file_path,
                                "page": page_num + 1,
                                "file_type": "pdf"
                            }
                        })
                doc.close()
                
            elif ext == ".docx":
                doc = DocxDocument(file_path)
                full_text = []
                for para in doc.paragraphs:
                    full_text.append(para.text)
                text = "\n".join(full_text)
                if text.strip():
                    documents.append({
                        "text": text,
                        "metadata": {
                            "source": filename,
                            "file_path": file_path,
                            "file_type": "docx"
                        }
                    })
                    
            elif ext == ".txt":
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                if text.strip():
                    documents.append({
                        "text": text,
                        "metadata": {
                            "source": filename,
                            "file_path": file_path,
                            "file_type": "txt"
                        }
                    })
            else:
                print(f"Format non supporté pour {file_path}")
        except Exception as e:
            print(f"Erreur lors de l'extraction de {file_path} : {e}")
            
        return documents

    def chunk_documents(
        self, 
        documents: List[Dict[str, Any]], 
        chunk_size: int = 500, 
        chunk_overlap: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Découpe les documents extraits en chunks avec chevauchement.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        chunks = []
        for doc in documents:
            texts = splitter.split_text(doc["text"])
            for idx, text in enumerate(texts):
                chunk_metadata = doc["metadata"].copy()
                chunk_metadata.update({
                    "chunk_id": f"{chunk_metadata['source']}_chunk_{idx}",
                    "chunk_index": idx,
                    "char_count": len(text)
                })
                chunks.append({
                    "text": text,
                    "metadata": chunk_metadata
                })
        return chunks

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        Génère les embeddings pour chaque chunk et les insère dans Qdrant.
        """
        if not chunks:
            print("Aucun chunk à indexer.")
            return False
            
        print(f"Indexation de {len(chunks)} chunks dans la collection '{self.collection_name}'...")
        
        texts = [chunk["text"] for chunk in chunks]
        
        try:
            # Génération des embeddings via Ollama
            embeddings_list = self.embeddings.embed_documents(texts)
            
            # Préparation des points pour Qdrant
            points = []
            for idx, (chunk, vector) in enumerate(zip(chunks, embeddings_list)):
                point_id = int(hash(chunk["metadata"]["chunk_id"]) % 1e15) # ID numérique unique
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "text": chunk["text"],
                            "metadata": chunk["metadata"],
                            "timestamp": time.time()
                        }
                    )
                )
            
            # Insertion par lots dans Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"Indexation réussie de {len(chunks)} points.")
            return True
        except Exception as e:
            print(f"Erreur d'indexation dans Qdrant : {e}")
            return False

    def run_pipeline(self, file_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> bool:
        """
        Exécute le pipeline complet pour un fichier donné.
        """
        print(f"Début du pipeline d'ingestion pour : {file_path}")
        raw_docs = self.extract_text(file_path)
        if not raw_docs:
            print("Extraction de texte vide ou échouée.")
            return False
            
        chunks = self.chunk_documents(raw_docs, chunk_size, chunk_overlap)
        success = self.index_chunks(chunks)
        return success

if __name__ == "__main__":
    # Test local rapide
    # pipeline = DocumentIngestionPipeline()
    # pipeline.run_pipeline("chemin/vers/un/doc.pdf")
    pass
