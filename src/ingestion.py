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

    def semantic_chunk_documents(self, documents: List[Dict[str, Any]], percentile_threshold: float = 90.0) -> List[Dict[str, Any]]:
        """
        Découpe les documents de manière sémantique en regroupant les phrases ayant un sens proche.
        """
        import re
        import numpy as np

        chunks = []
        for doc in documents:
            # 1. Découpage en phrases simples
            sentences = re.split(r'(?<=[.!?])\s+', doc["text"].strip())
            sentences = [s for s in sentences if len(s.strip()) > 10]
            
            if len(sentences) < 2:
                chunks.append({
                    "text": doc["text"],
                    "metadata": doc["metadata"]
                })
                continue
                
            # 2. Vectorisation de chaque phrase
            sentence_embeddings = self.embeddings.embed_documents(sentences)
            
            # 3. Calcul des distances de similarité cosinus
            similarities = []
            for i in range(len(sentence_embeddings) - 1):
                v1 = np.array(sentence_embeddings[i])
                v2 = np.array(sentence_embeddings[i+1])
                similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                similarities.append(similarity)
                
            distances = [1 - sim for sim in similarities]
            
            # 4. Calcul du seuil de coupure (percentile)
            threshold = np.percentile(distances, percentile_threshold)
            
            # 5. Regroupement des phrases en chunks
            current_chunk = [sentences[0]]
            chunk_idx = 0
            
            for i in range(len(sentences) - 1):
                if distances[i] > threshold:
                    chunk_text = " ".join(current_chunk)
                    chunk_metadata = doc["metadata"].copy()
                    chunk_metadata.update({
                        "chunk_id": f"{chunk_metadata['source']}_semantic_{chunk_idx}",
                        "chunk_index": chunk_idx,
                        "char_count": len(chunk_text),
                        "chunk_type": "semantic"
                    })
                    chunks.append({
                        "text": chunk_text,
                        "metadata": chunk_metadata
                    })
                    current_chunk = [sentences[i+1]]
                    chunk_idx += 1
                else:
                    current_chunk.append(sentences[i+1])
                    
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk_metadata = doc["metadata"].copy()
                chunk_metadata.update({
                    "chunk_id": f"{chunk_metadata['source']}_semantic_{chunk_idx}",
                    "chunk_index": chunk_idx,
                    "char_count": len(chunk_text),
                    "chunk_type": "semantic"
                })
                chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata
                })
                
        return chunks

    def index_chunks(self, chunks: List[Dict[str, Any]], progress_callback=None) -> bool:
        """
        Génère les embeddings pour chaque chunk et les insère dans Qdrant.
        """
        if not chunks:
            print("Aucun chunk à indexer.")
            return False
            
        print(f"Indexation de {len(chunks)} chunks dans la collection '{self.collection_name}'...")
        
        texts = [chunk["text"] for chunk in chunks]
        
        try:
            if progress_callback:
                progress_callback("embedding", 20, "Génération des embeddings...", len(chunks), 0)
                
            # Génération des embeddings via Ollama
            embeddings_list = self.embeddings.embed_documents(texts)
            
            if progress_callback:
                progress_callback("embedding", 100, "Embeddings générés.")
                progress_callback("qdrant", 20, "Insertion dans Qdrant...")
                
            # Préparation des points pour Qdrant
            import uuid
            points = []
            for idx, (chunk, vector) in enumerate(zip(chunks, embeddings_list)):
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["metadata"]["chunk_id"]))
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
            
            if progress_callback:
                progress_callback("completed", 100, "Indexation terminée avec succès.")
            return True
        except Exception as e:
            print(f"Erreur d'indexation dans Qdrant : {e}")
            if progress_callback:
                progress_callback("error", 0, f"Erreur Qdrant: {str(e)}")
            return False

    def run_pipeline(self, file_path: str, chunk_size: int = 500, chunk_overlap: int = 50, use_semantic: bool = False, progress_callback=None) -> bool:
        """
        Exécute le pipeline complet pour un fichier donné.
        """
        print(f"Début du pipeline d'ingestion pour : {file_path} (Sémantique : {use_semantic})")
        
        if progress_callback:
            progress_callback("extraction", 10, "Extraction du texte en cours...")
            
        raw_docs = self.extract_text(file_path)
        if not raw_docs:
            print("Extraction de texte vide ou échouée.")
            if progress_callback:
                progress_callback("error", 0, "Échec de l'extraction de texte.")
            return False
            
        if progress_callback:
            progress_callback("extraction", 100, "Texte extrait.", 0, len(raw_docs))
            progress_callback("chunking", 20, "Découpage du document...")
            
        if use_semantic:
            chunks = self.semantic_chunk_documents(raw_docs, percentile_threshold=90.0)
        else:
            chunks = self.chunk_documents(raw_docs, chunk_size, chunk_overlap)
            
        if progress_callback:
            progress_callback("chunking", 100, "Découpage terminé.", len(chunks), len(raw_docs))
            
        success = self.index_chunks(chunks, progress_callback=progress_callback)
        return success

if __name__ == "__main__":
    pipeline = DocumentIngestionPipeline()
    print("--- Ingestion TXT (Sémantique) ---")
    pipeline.run_pipeline("data/raw/politique_teletravail.txt", use_semantic=True)
    
    print("\n--- Ingestion DOCX (Sémantique) ---")
    pipeline.run_pipeline("data/raw/guide_securite.docx", use_semantic=True)
