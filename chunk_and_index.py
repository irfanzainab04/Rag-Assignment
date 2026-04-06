import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


load_dotenv()

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
UPSERT_BATCH_SIZE = 100

STRATEGIES = {
    "fixed": CharacterTextSplitter(chunk_size=512, chunk_overlap=64, separator=" "),
    "recursive": RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk corpus, create embeddings, and index to Pinecone.")
    parser.add_argument("--corpus", default="data/corpus_clean.json", help="Path to corpus JSON file.")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["fixed", "recursive"],
        choices=["fixed", "recursive"],
        help="Chunking strategies to build.",
    )
    parser.add_argument("--index-prefix", default="medical-rag", help="Pinecone index name prefix.")
    parser.add_argument("--region", default="us-east-1", help="Pinecone region.")
    parser.add_argument("--cloud", default="aws", help="Pinecone cloud provider.")
    return parser.parse_args()


def read_corpus(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_chunks(splitter, articles: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
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
                    "title": article["title"],
                    "source": article.get("source", "PubMed"),
                    "topic": article.get("topic", "unknown"),
                    "pubmed_id": str(article.get("id", ""))[:128],
                }
            )
    return chunks


def list_index_names(pc: Pinecone) -> List[str]:
    listed = pc.list_indexes()
    if hasattr(listed, "names"):
        return listed.names()
    return [item.name for item in listed]


def ensure_index(pc: Pinecone, index_name: str, cloud: str, region: str) -> None:
    if index_name in list_index_names(pc):
        return

    pc.create_index(
        name=index_name,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud=cloud, region=region),
    )


def upsert_chunks(index, model: SentenceTransformer, chunks: List[Dict[str, str]]) -> None:
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

    vectors: List[Tuple[str, List[float], Dict[str, str]]] = []
    for chunk, embedding in zip(chunks, embeddings):
        vectors.append(
            (
                chunk["id"],
                embedding.tolist(),
                {
                    "text": chunk["text"],
                    "title": chunk["title"],
                    "source": chunk["source"],
                    "topic": chunk["topic"],
                    "pubmed_id": chunk.get("pubmed_id", ""),
                },
            )
        )

    for i in range(0, len(vectors), UPSERT_BATCH_SIZE):
        index.upsert(vectors=vectors[i : i + UPSERT_BATCH_SIZE])


def save_bm25(chunks: List[Dict[str, str]], output_path: Path) -> None:
    tokenized = [chunk["text"].lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump({"bm25": bm25, "chunks": chunks}, handle)


def main() -> None:
    args = parse_args()
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise EnvironmentError("PINECONE_API_KEY is missing. Add it to .env or environment variables.")

    articles = read_corpus(Path(args.corpus))
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    pc = Pinecone(api_key=api_key)

    for strategy in args.strategies:
        splitter = STRATEGIES[strategy]
        chunks = build_chunks(splitter, articles)
        index_name = f"{args.index_prefix}-{strategy}"

        ensure_index(pc, index_name=index_name, cloud=args.cloud, region=args.region)
        index = pc.Index(index_name)

        upsert_chunks(index=index, model=embedder, chunks=chunks)
        save_bm25(chunks, Path(f"data/bm25_{strategy}.pkl"))
        print(f"Strategy '{strategy}': indexed {len(chunks)} chunks into {index_name}")


if __name__ == "__main__":
    main()