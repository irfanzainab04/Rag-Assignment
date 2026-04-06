import os
import json
import hashlib
import pickle
import time
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


load_dotenv()


def get_api_key(key_name: str) -> str:
    """Get API key from Streamlit secrets (Streamlit Cloud) or environment variables (local)."""
    try:
        import streamlit as st
        if key_name in st.secrets:
            return st.secrets[key_name]
    except (ImportError, Exception):
        pass
    
    env_value = os.getenv(key_name)
    if not env_value:
        raise EnvironmentError(f"{key_name} is missing. Add it to .env, environment variables, or Streamlit secrets.")
    return env_value

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

SEMANTIC_TOP_K = 30
BM25_TOP_K = 30
RRF_K = 60
FUSION_POOL = 30
FINAL_TOP_K = 5

STRATEGIES = {
    "fixed": CharacterTextSplitter(chunk_size=512, chunk_overlap=64, separator=" "),
    "recursive": RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100),
}

class Retriever:
    def __init__(self, strategy: str = "recursive", index_prefix: str = "medical-rag"):
        api_key = get_api_key("PINECONE_API_KEY")

        self.strategy = strategy
        self.index_name = f"{index_prefix}-{strategy}"
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)
        self.reranker = CrossEncoder(RERANKER_NAME)

        pc = Pinecone(api_key=api_key)
        self.index = pc.Index(self.index_name)
        self.bm25, self.bm25_chunks, self.chunk_map = self._load_bm25(strategy)

    @staticmethod
    def _load_bm25(strategy: str):
        bm25_path = Path(f"data/bm25_{strategy}.pkl")
        if not bm25_path.exists():
            return Retriever._build_bm25_from_corpus(strategy)

        with bm25_path.open("rb") as handle:
            payload = pickle.load(handle)

        bm25 = payload["bm25"]
        chunks = payload["chunks"]
        chunk_map = {chunk["id"]: chunk for chunk in chunks}
        return bm25, chunks, chunk_map

    @staticmethod
    def _build_bm25_from_corpus(strategy: str):
        """Fallback for cloud deploys where generated BM25 pickle is not committed."""
        corpus_path = Path("data/corpus_clean.json")
        if not corpus_path.exists():
            raise FileNotFoundError(
                f"BM25 cache missing and corpus not found at {corpus_path}. "
                "Ensure data/corpus_clean.json is included in the repository."
            )

        if strategy not in STRATEGIES:
            raise ValueError(f"Unsupported strategy '{strategy}'. Choose from {sorted(STRATEGIES)}")

        articles = json.loads(corpus_path.read_text(encoding="utf-8"))
        splitter = STRATEGIES[strategy]

        chunks: List[Dict[str, str]] = []
        for article in articles:
            text = f"{article['title']}. {article['abstract']}"
            article_uid = hashlib.sha1(
                f"{article.get('id', '')}|{article.get('title', '')}".encode("utf-8")
            ).hexdigest()[:16]
            for i, chunk in enumerate(splitter.split_text(text)):
                chunks.append(
                    {
                        "id": f"{article_uid}_{i}",
                        "text": chunk,
                        "title": article.get("title", "Untitled"),
                        "source": article.get("source", "PubMed"),
                        "topic": article.get("topic", "unknown"),
                        "pubmed_id": str(article.get("id", ""))[:128],
                    }
                )

        tokenized = [chunk["text"].lower().split() for chunk in chunks]
        bm25 = BM25Okapi(tokenized)
        chunk_map = {chunk["id"]: chunk for chunk in chunks}
        return bm25, chunks, chunk_map

    def _semantic_search(self, query: str, top_k: int = SEMANTIC_TOP_K) -> List[Dict[str, str]]:
        query_vec = self.embedder.encode(query).tolist()
        results = self.index.query(vector=query_vec, top_k=top_k, include_metadata=True)

        items: List[Dict[str, str]] = []
        for match in results["matches"]:
            meta = match.get("metadata") or {}
            items.append(
                {
                    "id": match["id"],
                    "text": meta.get("text", self.chunk_map.get(match["id"], {}).get("text", "")),
                    "title": meta.get("title", self.chunk_map.get(match["id"], {}).get("title", "Untitled")),
                    "source": meta.get("source", self.chunk_map.get(match["id"], {}).get("source", "PubMed")),
                    "topic": meta.get("topic", self.chunk_map.get(match["id"], {}).get("topic", "unknown")),
                    "score": float(match.get("score", 0.0)),
                    "source_leg": "semantic",
                }
            )
        return items

    def _bm25_search(self, query: str, top_k: int = BM25_TOP_K) -> List[Dict[str, str]]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_positions = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]

        items = []
        for idx in top_positions:
            chunk = self.bm25_chunks[idx]
            items.append({**chunk, "score": float(scores[idx]), "source_leg": "bm25"})
        return items

    @staticmethod
    def _rrf_fusion(semantic_results: List[Dict[str, str]], bm25_results: List[Dict[str, str]]) -> List[Dict[str, str]]:
        sem_rank = {item["id"]: rank for rank, item in enumerate(semantic_results, start=1)}
        bm_rank = {item["id"]: rank for rank, item in enumerate(bm25_results, start=1)}
        merged: Dict[str, Dict[str, str]] = {}
        all_ids = set(sem_rank) | set(bm_rank)

        for doc_id in all_ids:
            score = 0.0
            if doc_id in sem_rank:
                score += 1.0 / (RRF_K + sem_rank[doc_id])
                merged[doc_id] = next(item.copy() for item in semantic_results if item["id"] == doc_id)
                merged[doc_id]["source_leg"] = "semantic"
            if doc_id in bm_rank:
                score += 1.0 / (RRF_K + bm_rank[doc_id])
                if doc_id not in merged:
                    merged[doc_id] = next(item.copy() for item in bm25_results if item["id"] == doc_id)
                    merged[doc_id]["source_leg"] = "bm25"
                else:
                    merged[doc_id]["source_leg"] = "both"
            merged[doc_id]["rrf_score"] = score

        return sorted(merged.values(), key=lambda item: -item["rrf_score"])

    def _rerank(self, query: str, candidates: List[Dict[str, str]], top_k: int = FINAL_TOP_K) -> List[Dict[str, str]]:
        if not candidates:
            return []
        pairs = [(query, item["text"]) for item in candidates[:FUSION_POOL]]
        scores = self.reranker.predict(pairs)
        for i, score in enumerate(scores):
            candidates[i]["rerank_score"] = float(score)
        return sorted(candidates[:FUSION_POOL], key=lambda item: -item["rerank_score"])[:top_k]

    def retrieve_semantic_only(self, query: str, top_k: int = FINAL_TOP_K) -> Dict[str, object]:
        t0 = time.perf_counter()
        chunks = self._semantic_search(query, top_k=top_k)
        return {
            "query": query,
            "strategy": self.strategy,
            "mode": "semantic_only",
            "chunks": chunks[:top_k],
            "retrieval_time": round(time.perf_counter() - t0, 4),
        }

    def retrieve_hybrid_reranked(self, query: str, top_k: int = FINAL_TOP_K, rerank: bool = True) -> Dict[str, object]:
        t0 = time.perf_counter()
        semantic_results = self._semantic_search(query, top_k=SEMANTIC_TOP_K)
        bm25_results = self._bm25_search(query, top_k=BM25_TOP_K)
        t_retrieval = time.perf_counter()

        fused = self._rrf_fusion(semantic_results, bm25_results)
        t_fusion = time.perf_counter()

        if rerank:
            final_chunks = self._rerank(query, fused, top_k=top_k)
            mode = "hybrid_reranked"
        else:
            final_chunks = fused[:top_k]
            mode = "hybrid_rrf"

        t_end = time.perf_counter()
        return {
            "query": query,
            "strategy": self.strategy,
            "mode": mode,
            "chunks": final_chunks,
            "timing": {
                "retrieval_time": round(t_retrieval - t0, 4),
                "fusion_time": round(t_fusion - t_retrieval, 4),
                "rerank_time": round(t_end - t_fusion, 4),
                "total_time": round(t_end - t0, 4),
            },
            "debug": {
                "semantic_candidates": len(semantic_results),
                "bm25_candidates": len(bm25_results),
                "fused_candidates": len(fused),
                "final_returned": len(final_chunks),
            },
        }

    def retrieve(self, query: str, mode: str = "hybrid_reranked", top_k: int = FINAL_TOP_K) -> Dict[str, object]:
        allowed_modes = {"semantic_only", "hybrid_rrf", "hybrid_reranked"}
        if mode not in allowed_modes:
            raise ValueError(f"Invalid mode '{mode}'. Choose from {sorted(allowed_modes)}")

        if mode == "semantic_only":
            return self.retrieve_semantic_only(query, top_k=top_k)
        if mode == "hybrid_rrf":
            return self.retrieve_hybrid_reranked(query, top_k=top_k, rerank=False)
        return self.retrieve_hybrid_reranked(query, top_k=top_k, rerank=True)


_RETRIEVER_CACHE: Dict[str, Retriever] = {}


def get_retriever(strategy: str = "recursive") -> Retriever:
    if strategy not in _RETRIEVER_CACHE:
        _RETRIEVER_CACHE[strategy] = Retriever(strategy=strategy)
    return _RETRIEVER_CACHE[strategy]


def hybrid_retrieve(query: str, strategy: str = "recursive", mode: str = "hybrid_reranked") -> List[Dict[str, str]]:
    retriever = get_retriever(strategy)
    result = retriever.retrieve(query=query, mode=mode, top_k=FINAL_TOP_K)
    return result["chunks"]