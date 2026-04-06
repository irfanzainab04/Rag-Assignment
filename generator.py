import os
import time
import re
from typing import Dict, List

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


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


HF_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
FALLBACK_MODELS = [HF_MODEL_ID]
_HF_CLIENT: InferenceClient | None = None
_HF_DISABLED = False


def _build_prompt(query: str, chunks: List[Dict[str, str]]) -> str:
    context = "\n\n".join(
        f"[Chunk {i + 1}] Source: {chunk.get('source', 'PubMed')} | Title: {chunk.get('title', 'Untitled')}\n{chunk['text']}"
        for i, chunk in enumerate(chunks)
    )

    return f"""You are a careful medical information assistant.
Use only the provided context. Do not add outside facts.
If the context does not clearly support an answer, say: "I do not have enough evidence in the retrieved context."
Write a short answer in 3-4 sentences.
Use cautious language such as "the context suggests" or "the retrieved evidence indicates".
When possible, mention the chunk numbers that support the answer.

Context:
{context}

Question: {query}
Answer:"""


def _local_extractive_answer(query: str, chunks: List[Dict[str, str]]) -> str:
    query_terms = {term for term in re.findall(r"[a-zA-Z]{4,}", query.lower())}
    scored_chunks = []
    for chunk in chunks:
        text = chunk.get("text", "")
        text_terms = set(re.findall(r"[a-zA-Z]{4,}", text.lower()))
        overlap = len(query_terms & text_terms)
        scored_chunks.append((overlap, text, chunk))

    scored_chunks.sort(key=lambda item: (-item[0], -len(item[1])))

    selected_sentences = []
    for overlap, text, chunk in scored_chunks[:3]:
        if not text:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sentence in sentences:
            sentence_terms = set(re.findall(r"[a-zA-Z]{4,}", sentence.lower()))
            if overlap > 0 and sentence_terms & query_terms:
                selected_sentences.append(
                    f"[{chunk.get('title', 'Untitled')}] {sentence.strip()}"
                )
                break
        if len(selected_sentences) >= 3:
            break

    if not selected_sentences and chunks:
        first_chunk = chunks[0]
        fallback_text = first_chunk.get("text", "")
        first_sentence = re.split(r"(?<=[.!?])\s+", fallback_text)[0].strip()
        selected_sentences.append(f"[{first_chunk.get('title', 'Untitled')}] {first_sentence}")

    if not selected_sentences:
        return "I do not have enough evidence in the retrieved context."

    joined = " ".join(selected_sentences[:3])
    return (
        "The retrieved context suggests the following: "
        + joined
        + " This answer is based only on the retrieved evidence."
    )


def _get_hf_client() -> InferenceClient:
    global _HF_CLIENT
    if _HF_CLIENT is None:
        token = get_api_key("HF_API_TOKEN")
        _HF_CLIENT = InferenceClient(token=token)
    return _HF_CLIENT


def _call_hf_api(
    prompt: str,
    model_id: str,
    max_new_tokens: int = 400,
    temperature: float = 0.2,
) -> str:
    global _HF_DISABLED
    if _HF_DISABLED:
        raise RuntimeError("HF generation disabled after quota error")
    client = _get_hf_client()
    messages = [
        {"role": "system", "content": "You are a precise medical QA assistant."},
        {"role": "user", "content": prompt},
    ]
    response = client.chat_completion(
        messages,
        model=model_id,
        max_tokens=max_new_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


class Generator:
    def __init__(self, model_name: str = HF_MODEL_ID):
        self.model_name = model_name

    def generate(self, query: str, chunks: List[Dict[str, str]], max_new_tokens: int = 400) -> Dict[str, object]:
        global _HF_DISABLED
        prompt = _build_prompt(query, chunks)
        start = time.perf_counter()
        last_error = None

        models_to_try = [self.model_name] + [model for model in FALLBACK_MODELS if model != self.model_name]
        for model in models_to_try:
            try:
                answer = _call_hf_api(
                    prompt,
                    model_id=model,
                    max_new_tokens=max_new_tokens,
                )
                return {
                    "answer": answer.strip(),
                    "model": model,
                    "provider": "hf-inference",
                    "query": query,
                    "num_chunks_used": len(chunks),
                    "generation_time": round(time.perf_counter() - start, 4),
                    "status": "success",
                }
            except Exception as error:  # noqa: BLE001
                last_error = error
                message = str(error).lower()
                if "402" in message or "depleted" in message or "quota" in message:
                    _HF_DISABLED = True

        answer = _local_extractive_answer(query, chunks)
        return {
            "answer": answer,
            "model": f"local-fallback-after-hf-error:{self.model_name}",
            "provider": "local-fallback",
            "query": query,
            "num_chunks_used": len(chunks),
            "generation_time": round(time.perf_counter() - start, 4),
            "status": "fallback",
        }


_GENERATOR_CACHE: Generator | None = None


def get_generator() -> Generator:
    global _GENERATOR_CACHE
    if _GENERATOR_CACHE is None:
        _GENERATOR_CACHE = Generator()
    return _GENERATOR_CACHE


def generate_answer(query: str, chunks: List[Dict[str, str]]) -> str:
    result = get_generator().generate(query=query, chunks=chunks)
    return str(result["answer"]).strip()