import sqlite3
import json
import time
import os
from typing import Dict, Any, List

DB_PATH = os.path.join("data", "monitoring.db")

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Table for query metrics (No PII / Full Text)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS queries (
        query_id TEXT PRIMARY KEY,
        timestamp REAL,
        retrieval_mode TEXT,
        top_k INTEGER,
        reranking_enabled INTEGER,
        retrieval_latency REAL,
        hybrid_latency REAL,
        reranking_latency REAL,
        llm_latency REAL,
        total_latency REAL,
        candidates_count INTEGER,
        final_sources_count INTEGER,
        status TEXT,
        error_type TEXT,
        rag_trace TEXT
    )
    ''')
    
    # Table for logs
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL,
        level TEXT,
        service TEXT,
        message TEXT,
        query_id TEXT,
        duration REAL
    )
    ''')
    
    conn.commit()
    conn.close()

class MonitoringService:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        init_db()
        
    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def record_query(self, query_id: str, metrics: Dict[str, Any], trace: Dict[str, Any] = None):
        """Asynchronous recording of a query. Should be called via BackgroundTasks."""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Privacy scrub on trace: remove any "text" fields from sources if present
            scrubbed_trace = json.dumps(self._scrub_trace(trace)) if trace else None
            
            cursor.execute('''
            INSERT OR REPLACE INTO queries (
                query_id, timestamp, retrieval_mode, top_k, reranking_enabled,
                retrieval_latency, hybrid_latency, reranking_latency, llm_latency, total_latency,
                candidates_count, final_sources_count, status, error_type, rag_trace
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                query_id,
                time.time(),
                metrics.get("retrieval_mode", "hybrid"),
                metrics.get("top_k", 5),
                int(metrics.get("reranking", True)),
                metrics.get("latencies", {}).get("retrieval", 0.0),
                metrics.get("latencies", {}).get("hybrid", 0.0),
                metrics.get("latencies", {}).get("reranking", 0.0),
                metrics.get("latencies", {}).get("llm", 0.0),
                metrics.get("latencies", {}).get("total", 0.0),
                metrics.get("candidates_count", 0),
                metrics.get("final_sources_count", 0),
                metrics.get("status", "SUCCESS"),
                metrics.get("error_type", None),
                scrubbed_trace
            ))
            
            conn.commit()
        except Exception as e:
            print(f"[MonitoringService] Error recording query {query_id}: {e}")
        finally:
            if conn:
                conn.close()

    def record_log(self, level: str, service: str, message: str, query_id: str = None, duration: float = None):
        """Asynchronous log recording."""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO logs (timestamp, level, service, message, query_id, duration)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (time.time(), level.upper(), service, message, query_id, duration))
            conn.commit()
        except Exception as e:
            print(f"[MonitoringService] Error recording log: {e}")
        finally:
            if conn:
                conn.close()

    def get_metrics(self, start_timestamp: float) -> Dict[str, Any]:
        """Calculates aggregations like P50/P95 for a specific time window."""
        conn = None
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM queries WHERE timestamp >= ?", (start_timestamp,))
            rows = cursor.fetchall()
            
            total = len(rows)
            failed = sum(1 for r in rows if r["status"] == "ERROR")
            no_context = sum(1 for r in rows if r["status"] in ["NO_CONTEXT", "SHORT_CIRCUIT"] or (r["status"] != "ERROR" and (r["final_sources_count"] is None or r["final_sources_count"] == 0)))
            successful = sum(1 for r in rows if r["status"] != "ERROR" and r["final_sources_count"] is not None and r["final_sources_count"] > 0)
            
            success_rate = round((successful / total * 100) if total else 0.0, 1)
            error_rate = round((failed / total * 100) if total else 0.0, 1)
            no_context_rate = round((no_context / total * 100) if total else 0.0, 1)

            # Calculate latencies
            total_latencies = sorted([r["total_latency"] for r in rows if r["total_latency"] is not None])
            
            def p(pct, lst):
                if not lst: return 0.0
                idx = int(len(lst) * pct)
                return round(lst[min(idx, len(lst)-1)], 2)
            
            def avg(key):
                vals = [r[key] for r in rows if r[key] is not None]
                return round(sum(vals)/len(vals), 2) if vals else 0.0

            latency_stats = {
                "avg": avg("total_latency"),
                "p50": p(0.50, total_latencies),
                "p95": p(0.95, total_latencies),
                "p99": p(0.99, total_latencies)
            }
            
            retrieval_stats = {
                "vector": sum(1 for r in rows if r["retrieval_mode"] == "vector"),
                "keyword": sum(1 for r in rows if r["retrieval_mode"] == "keyword"),
                "hybrid": sum(1 for r in rows if r["retrieval_mode"] == "hybrid"),
                "avg_candidates": avg("candidates_count"),
                "avg_final": avg("final_sources_count")
            }
            
            reranking_stats = {
                "enabled_rate": round((sum(1 for r in rows if r["reranking_enabled"] == 1) / total * 100) if total else 0, 1),
                "avg_latency": avg("reranking_latency")
            }
            
            return {
                "queries": {
                    "total": total,
                    "successful": successful,
                    "failed": failed,
                    "no_context": no_context,
                    "success_rate": success_rate,
                    "error_rate": error_rate,
                    "no_context_rate": no_context_rate
                },
                "latency": latency_stats,
                "latency_breakdown": {
                    "retrieval": avg("retrieval_latency"),
                    "hybrid": avg("hybrid_latency"),
                    "reranking": avg("reranking_latency"),
                    "llm": avg("llm_latency"),
                    "total": latency_stats["avg"]
                },
                "retrieval": retrieval_stats,
                "reranking": reranking_stats
            }
        except Exception as e:
            print(f"[MonitoringService] Error in get_metrics: {e}")
            return {}
        finally:
            if conn:
                conn.close()

    def get_logs(self, limit: int = 50, level: str = None) -> List[Dict[str, Any]]:
        conn = None
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = "SELECT * FROM logs"
            params = []
            if level and level != "ALL":
                query += " WHERE level = ?"
                params.append(level)
                
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            print(f"[MonitoringService] Error in get_logs: {e}")
            return []
        finally:
            if conn:
                conn.close()
        
    def get_trace(self, query_id: str) -> Dict[str, Any]:
        conn = None
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM queries WHERE query_id = ?", (query_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
                
            result = dict(row)
            if result["rag_trace"]:
                try:
                    result["rag_trace"] = json.loads(result["rag_trace"])
                except:
                    result["rag_trace"] = {}
            return result
        except Exception as e:
            print(f"[MonitoringService] Error in get_trace: {e}")
            return None
        finally:
            if conn:
                conn.close()

    def _scrub_trace(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        """Deep clean of PII/text from trace."""
        if not trace: return {}
        clean = json.loads(json.dumps(trace)) # Deep copy
        
        # Supprimer la query originale si présente
        if "query" in clean:
            clean["query"] = "[REDACTED]"
            
        # Parcourir les sources pour supprimer le texte
        if "steps" in clean:
            for step in clean["steps"]:
                if "sources" in step:
                    for src in step["sources"]:
                        if "text" in src:
                            src["text"] = "[REDACTED]"
        return clean

monitoring_service = MonitoringService()
