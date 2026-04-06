import os
import re
import time
from typing import Dict, List, Tuple

import numpy as np
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
HF_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
FALLBACK_MODELS = [HF_MODEL_ID]
embedder = SentenceTransformer(EMBED_MODEL_NAME)
_HF_CLIENT: InferenceClient | None = None
_HF_DISABLED = False


STOPWORDS = {
    "the", "and", "with", "that", "this", "from", "were", "have", "has", "had",
    "for", "into", "their", "there", "about", "which", "when", "where", "what",
    "how", "why", "are", "was", "were", "been", "being", "will", "would", "could",
    "should", "can", "may", "might", "also", "than", "then", "them", "they", "this",
    "these", "those", "such", "more", "most", "less", "very", "patient", "patients",
}


def _get_hf_client() -> InferenceClient:
    global _HF_CLIENT
    if _HF_CLIENT is None:
        token = os.getenv("HF_API_TOKEN")
        if not token:
            raise EnvironmentError("HF_API_TOKEN is missing. Add it to .env or environment variables.")
        _HF_CLIENT = InferenceClient(token=token)
    return _HF_CLIENT


def call_llm(prompt: str, max_new_tokens: int = 300, temperature: float = 0.1) -> str:
    global _HF_DISABLED
    if _HF_DISABLED:
        raise RuntimeError("HF generation disabled after quota error")

    last_error = None
    for model_id in FALLBACK_MODELS:
        try:
            client = _get_hf_client()
            response = client.chat_completion(
                [
                    {"role": "system", "content": "You are an evaluator for RAG outputs."},
                    {"role": "user", "content": prompt},
                ],
                model=model_id,
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as error:  # noqa: BLE001
            last_error = error
            message = str(error).lower()
            if "402" in message or "depleted" in message or "quota" in message:
                _HF_DISABLED = True

    raise RuntimeError(f"HF router API failed for all fallback models/providers. Last error: {last_error}")


def _extract_numbered_items(text: str, max_items: int = 10) -> List[str]:
    items = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^\d+[\).:-]?\s*(.+)$", line)
        if match:
            items.append(match.group(1).strip())
    return items[:max_items]


def _tokenize(text: str) -> List[str]:
    return [token for token in re.findall(r"[a-zA-Z]{4,}", text.lower()) if token not in STOPWORDS]


def _local_claim_extraction(answer: str) -> List[str]:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", answer) if sentence.strip()]
    if not sentences:
        return [answer.strip()] if answer.strip() else []
    claims = []
    for sentence in sentences:
        cleaned = sentence.strip()
        if cleaned and len(cleaned) > 20:
            claims.append(cleaned)
    return claims[:8]


def _local_support_verdict(claim: str, context: str) -> str:
    claim_terms = set(_tokenize(claim))
    context_terms = set(_tokenize(context))
    if not claim_terms:
        return "UNSUPPORTED"
    overlap = len(claim_terms & context_terms)
    ratio = overlap / max(len(claim_terms), 1)
    return "SUPPORTED" if ratio >= 0.35 or overlap >= 3 else "UNSUPPORTED"


def _local_relevancy_questions(query: str, answer: str) -> List[str]:
    query_core = query.rstrip("? .")
    return [
        f"What does the answer say about {query_core}?",
        f"How is {query_core.lower()} described in the retrieved context?",
        f"What are the main details in the answer about {query_core.lower()}?",
    ]


def evaluate_faithfulness(answer: str, context_chunks: List[Dict[str, str]]) -> Tuple[float, List[Dict[str, str]]]:
    context = "\n".join(chunk["text"] for chunk in context_chunks)

    extract_prompt = f"""Extract only the explicit factual claims from the answer as a numbered list.
Each item must be one short, verifiable statement from the answer.
Do not rewrite or infer beyond what the answer says.

Return exactly this format:
1. claim text
2. claim text

Answer:
{answer}"""
    try:
        claims_text = call_llm(extract_prompt, max_new_tokens=256, temperature=0.0)
        claims = _extract_numbered_items(claims_text, max_items=12)
    except Exception:
        claims = _local_claim_extraction(answer)

    if not claims:
        claims = _local_claim_extraction(answer)

    supported_count = 0
    results: List[Dict[str, str]] = []

    for claim in claims:
        verify_prompt = f"""Check whether the claim is directly supported by the context.
    Output exactly one token: SUPPORTED or UNSUPPORTED.
    Be strict: if the context does not clearly support the claim, mark UNSUPPORTED.

    Context:
    {context[:2500]}

    Claim:
    {claim}"""
        try:
            verdict_text = call_llm(verify_prompt, max_new_tokens=5, temperature=0.0).upper()
            verdict = "SUPPORTED" if "SUPPORTED" in verdict_text and "UNSUPPORTED" not in verdict_text else "UNSUPPORTED"
        except Exception:
            verdict = _local_support_verdict(claim, context)
        if verdict == "SUPPORTED":
            supported_count += 1
        results.append({"claim": claim, "verdict": verdict})

    score = supported_count / len(claims) if claims else 0.0
    return round(score, 3), results


def evaluate_relevancy(query: str, answer: str) -> Tuple[float, List[str], List[float]]:
    gen_prompt = f"""Generate exactly 3 diverse questions that this answer could respond to.
Use numbered format:
1. ...
2. ...
3. ...

Keep each question close to the original topic.

Answer:
{answer}"""
    try:
        questions_text = call_llm(gen_prompt, max_new_tokens=180, temperature=0.2)
        questions = _extract_numbered_items(questions_text, max_items=3)
    except Exception:
        questions = _local_relevancy_questions(query, answer)

    if len(questions) < 3:
        fallback_questions = _local_relevancy_questions(query, answer)
        questions = (questions + fallback_questions)[:3]

    query_emb = embedder.encode([query])
    question_embs = embedder.encode(questions)
    similarities = cosine_similarity(query_emb, question_embs)[0]
    similarity_list = [round(float(score), 4) for score in similarities.tolist()]
    return round(float(np.mean(similarities)), 3), questions, similarity_list


class Evaluator:
    def evaluate_faithfulness(self, answer: str, chunks: List[Dict[str, str]]) -> Dict[str, object]:
        start = time.perf_counter()
        score, claims = evaluate_faithfulness(answer, chunks)
        supported = sum(1 for item in claims if item.get("verdict") == "SUPPORTED")
        return {
            "faithfulness_score": score,
            "claims": claims,
            "num_claims": len(claims),
            "num_supported": supported,
            "evaluation_time": round(time.perf_counter() - start, 4),
        }

    def evaluate_relevancy(self, query: str, answer: str) -> Dict[str, object]:
        start = time.perf_counter()
        score, questions, similarities = evaluate_relevancy(query, answer)
        return {
            "relevancy_score": score,
            "generated_questions": questions,
            "similarities": similarities,
            "evaluation_time": round(time.perf_counter() - start, 4),
        }

    def evaluate(self, query: str, answer: str, chunks: List[Dict[str, str]]) -> Dict[str, object]:
        faith = self.evaluate_faithfulness(answer, chunks)
        relev = self.evaluate_relevancy(query, answer)
        combined = round((faith["faithfulness_score"] + relev["relevancy_score"]) / 2, 4)
        return {
            "faithfulness": faith,
            "relevancy": relev,
            "combined_score": combined,
        }


_EVALUATOR_CACHE: Evaluator | None = None


def get_evaluator() -> Evaluator:
    global _EVALUATOR_CACHE
    if _EVALUATOR_CACHE is None:
        _EVALUATOR_CACHE = Evaluator()
    return _EVALUATOR_CACHE