import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Set

from pymed import PubMed


DEFAULT_TOPICS = [
    "diabetes treatment",
    "hypertension management",
    "cancer immunotherapy",
    "COVID-19 symptoms",
    "heart disease prevention",
    "mental health depression",
    "antibiotic resistance",
    "vaccine efficacy",
    "Alzheimer disease",
    "obesity treatment",
]


def normalize_pubmed_id(raw_id: object) -> str:
    """Convert `article.pubmed_id` values from pymed into a plain string id."""
    return str(raw_id).strip()


def fetch_articles(
    client: PubMed,
    topics: Iterable[str],
    max_per_topic: int,
    min_abstract_len: int,
) -> List[Dict[str, str]]:
    articles: List[Dict[str, str]] = []
    seen_ids: Set[str] = set()

    for topic in topics:
        print(f"Fetching topic: {topic}")
        for article in client.query(topic, max_results=max_per_topic):
            article_id = normalize_pubmed_id(article.pubmed_id)
            if article_id in seen_ids:
                continue

            abstract = (article.abstract or "").strip()
            if len(abstract) < min_abstract_len:
                continue

            seen_ids.add(article_id)
            articles.append(
                {
                    "id": article_id,
                    "title": (article.title or "Untitled").strip(),
                    "abstract": abstract,
                    "source": "PubMed",
                    "topic": topic,
                }
            )

    return articles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape PubMed abstracts for medical RAG corpus.")
    parser.add_argument(
        "--email",
        required=True,
        help="Your email for PubMed API identification.",
    )
    parser.add_argument(
        "--max-per-topic",
        type=int,
        default=30,
        help="Maximum results to fetch per topic.",
    )
    parser.add_argument(
        "--min-abstract-len",
        type=int,
        default=120,
        help="Minimum abstract length to keep.",
    )
    parser.add_argument(
        "--output",
        default="data/corpus_clean.json",
        help="Output JSON file path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pubmed = PubMed(tool="MedicalRAG", email=args.email)
    articles = fetch_articles(
        client=pubmed,
        topics=DEFAULT_TOPICS,
        max_per_topic=args.max_per_topic,
        min_abstract_len=args.min_abstract_len,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(articles, indent=2), encoding="utf-8")

    print(f"Saved {len(articles)} unique articles to {output_path}")


if __name__ == "__main__":
    main()