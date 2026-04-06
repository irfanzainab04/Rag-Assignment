import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

from evaluator import get_evaluator
from generator import get_generator
from retriever import get_retriever


load_dotenv()

TEST_QUERIES = [
    "What are common symptoms of Type 2 diabetes?",
    "How is hypertension usually managed in adults?",
    "What side effects are associated with chemotherapy?",
    "How do COVID-19 vaccines reduce severe illness?",
    "What factors are linked with Alzheimer's disease?",
    "How can obesity be treated in clinical practice?",
    "What are risks of antibiotic resistance?",
    "How does cancer immunotherapy work?",
    "What lifestyle changes help prevent heart disease?",
    "What are typical signs of depression?",
]

CONFIGS = [
    {"chunking": "fixed", "retrieval": "semantic_only", "label": "fixed_semantic_only"},
    {"chunking": "fixed", "retrieval": "hybrid_reranked", "label": "fixed_hybrid_reranked"},
    {"chunking": "recursive", "retrieval": "semantic_only", "label": "recursive_semantic_only"},
    {"chunking": "recursive", "retrieval": "hybrid_reranked", "label": "recursive_hybrid_reranked"},
]


def evaluate_single_query(query: str, strategy: str, retrieval_mode: str) -> Dict:
    retriever = get_retriever(strategy)
    generator = get_generator()
    evaluator = get_evaluator()

    retrieval_start = time.perf_counter()
    ret_result = retriever.retrieve(query=query, mode=retrieval_mode)
    chunks = ret_result["chunks"]
    retrieval_time = time.perf_counter() - retrieval_start

    generation_start = time.perf_counter()
    gen_result = generator.generate(query=query, chunks=chunks)
    answer = str(gen_result["answer"])
    generation_time = time.perf_counter() - generation_start

    faith_start = time.perf_counter()
    faith_result = evaluator.evaluate_faithfulness(answer, chunks)
    faithfulness_score = float(faith_result["faithfulness_score"])
    claims = list(faith_result["claims"])
    faith_eval_time = time.perf_counter() - faith_start

    relev_start = time.perf_counter()
    relev_result = evaluator.evaluate_relevancy(query, answer)
    relevancy_score = float(relev_result["relevancy_score"])
    generated_questions = list(relev_result["generated_questions"])
    similarities = list(relev_result["similarities"])
    relev_eval_time = time.perf_counter() - relev_start

    total_time = retrieval_time + generation_time + faith_eval_time + relev_eval_time

    return {
        "query": query,
        "answer": answer,
        "model": gen_result.get("model", "unknown"),
        "retrieval_debug": ret_result.get("debug", {}),
        "retrieved_chunks": chunks,
        "faithfulness": {
            "score": faithfulness_score,
            "claim_verification": claims,
            "time_sec": round(faith_eval_time, 3),
        },
        "relevancy": {
            "score": relevancy_score,
            "generated_questions": generated_questions,
            "similarities": similarities,
            "time_sec": round(relev_eval_time, 3),
        },
        "latency": {
            "retrieval_sec": round(retrieval_time, 3),
            "generation_sec": round(generation_time, 3),
            "total_sec": round(total_time, 3),
        },
    }


def summarize_runs(run_details: List[Dict]) -> Dict:
    if not run_details:
        return {
            "faithfulness_mean": 0.0,
            "relevancy_mean": 0.0,
            "retrieval_time_mean_sec": 0.0,
            "generation_time_mean_sec": 0.0,
            "total_time_mean_sec": 0.0,
        }

    faith = [run["faithfulness"]["score"] for run in run_details]
    rel = [run["relevancy"]["score"] for run in run_details]
    retrieval_times = [run["latency"]["retrieval_sec"] for run in run_details]
    generation_times = [run["latency"]["generation_sec"] for run in run_details]
    total_times = [run["latency"]["total_sec"] for run in run_details]

    return {
        "faithfulness_mean": round(statistics.mean(faith), 3),
        "relevancy_mean": round(statistics.mean(rel), 3),
        "retrieval_time_mean_sec": round(statistics.mean(retrieval_times), 3),
        "generation_time_mean_sec": round(statistics.mean(generation_times), 3),
        "total_time_mean_sec": round(statistics.mean(total_times), 3),
    }


def run_ablation(test_queries: List[str], selected_configs: List[Dict], continue_on_error: bool) -> Dict:
    results = {"configs": []}

    for config in selected_configs:
        print(
            f"Running config: chunking={config['chunking']} retrieval={config['retrieval']} ({config['label']})"
        )

        query_runs = []
        for i, query in enumerate(test_queries, start=1):
            print(f"  [{i}/{len(test_queries)}] {query}")
            try:
                query_runs.append(
                    evaluate_single_query(
                        query=query,
                        strategy=config["chunking"],
                        retrieval_mode=config["retrieval"],
                    )
                )
            except Exception as error:  # noqa: BLE001 - keep long runs alive when enabled
                print(f"  [WARN] Query failed: {error}")
                if continue_on_error:
                    continue
                raise

        summary = summarize_runs(query_runs)
        results["configs"].append(
            {
                "label": config["label"],
                "chunking": config["chunking"],
                "retrieval": config["retrieval"],
                "summary": summary,
                "query_runs": query_runs,
            }
        )

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation study for the medical RAG system.")
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Evaluate only the first N queries (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional subset of config labels to run. Example: recursive_hybrid_reranked",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining queries/configs if one query fails.",
    )
    parser.add_argument(
        "--output",
        default="data/evaluation_results/ablation_summary.json",
        help="Path to write JSON results.",
    )
    return parser.parse_args()


def save_results(payload: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()

    queries = TEST_QUERIES[: args.max_queries] if args.max_queries else TEST_QUERIES
    configs = CONFIGS
    if args.labels:
        allowed = set(args.labels)
        configs = [cfg for cfg in CONFIGS if cfg["label"] in allowed]
        if not configs:
            raise ValueError(f"No configs matched labels: {args.labels}")

    output_file = Path(args.output)
    all_results = run_ablation(
        test_queries=queries,
        selected_configs=configs,
        continue_on_error=args.continue_on_error,
    )
    save_results(all_results, output_file)
    print(f"Saved evaluation results to {output_file}")


if __name__ == "__main__":
    main()