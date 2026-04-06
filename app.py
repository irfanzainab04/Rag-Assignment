import time
from typing import Dict

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from evaluator import get_evaluator
from generator import get_generator
from retriever import get_retriever

st.set_page_config(page_title="Medical RAG QA", page_icon="+", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');
.stApp { font-family: 'Manrope', sans-serif; }
.hero { padding: 1.2rem 1.4rem; border-radius: 14px; background: linear-gradient(130deg, #073b3a, #0b6e4f 55%, #2a9d8f); color: #eef7f4; margin-bottom: 1rem; }
.hero h1 { margin: 0 0 .2rem 0; font-size: 2rem; letter-spacing: .2px; }
.hero p { margin: 0; opacity: 0.92; }
.metric-box { padding: .7rem .9rem; border: 1px solid #d4e8df; border-radius: 10px; background: #f7fcfa; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>Medical RAG QA Assistant</h1>
  <p>Hybrid Search (BM25 + Semantic), RRF Fusion, Re-ranking, and LLM-as-a-Judge evaluation.</p>
</div>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_retriever(strategy: str):
    return get_retriever(strategy)


@st.cache_resource
def load_generator():
    return get_generator()


@st.cache_resource
def load_evaluator():
    return get_evaluator()


def color_for_score(score: float) -> str:
    if score >= 0.7:
        return "#1c9a5f"
    if score >= 0.4:
        return "#bd7d00"
    return "#b42318"

query = st.text_input("Ask a medical question:", placeholder="e.g. What are treatments for Type 2 diabetes?")
with st.sidebar:
    st.markdown("### Settings")
    strategy = st.selectbox("Chunking strategy", options=["recursive", "fixed"], index=0)
    mode = st.selectbox("Retrieval mode", options=["hybrid_reranked", "hybrid_rrf", "semantic_only"], index=0)
    top_k = st.slider("Top-K chunks", min_value=3, max_value=10, value=5)
    run_eval = st.checkbox("Run LLM-as-a-Judge", value=True)
    st.markdown("---")
    st.caption("Domain: PubMed abstracts")
    st.caption("Embedding: all-MiniLM-L6-v2")
    st.caption("Reranker: ms-marco-MiniLM-L-6-v2")

if st.button("Get Answer") and query:
    with st.spinner("Retrieving and generating..."):
        retriever = load_retriever(strategy)
        generator = load_generator()

        t0 = time.perf_counter()
        ret_result: Dict[str, object] = retriever.retrieve(query=query, mode=mode, top_k=top_k)
        chunks = ret_result["chunks"]
        retrieval_sec = time.perf_counter() - t0

        t1 = time.perf_counter()
        gen_result = generator.generate(query=query, chunks=chunks)
        answer = gen_result["answer"]
        generation_sec = time.perf_counter() - t1

    faith_score = 0.0
    claims = []
    relev_score = 0.0
    questions = []
    relev_sims = []
    faith_eval_sec = 0.0
    relev_eval_sec = 0.0

    if run_eval:
        with st.spinner("Evaluating faithfulness and relevancy..."):
            evaluator = load_evaluator()

            t2 = time.perf_counter()
            faith_result = evaluator.evaluate_faithfulness(answer, chunks)
            faith_eval_sec = time.perf_counter() - t2

            t3 = time.perf_counter()
            relev_result = evaluator.evaluate_relevancy(query, answer)
            relev_eval_sec = time.perf_counter() - t3

            faith_score = float(faith_result["faithfulness_score"])
            claims = list(faith_result["claims"])
            relev_score = float(relev_result["relevancy_score"])
            questions = list(relev_result["generated_questions"])
            relev_sims = list(relev_result["similarities"])

    total_time = retrieval_sec + generation_sec + faith_eval_sec + relev_eval_sec

    st.subheader("Answer")
    st.write(answer)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Faithfulness", f"{faith_score:.0%}" if run_eval else "Skipped")
    col2.metric("Relevancy", f"{relev_score:.0%}" if run_eval else "Skipped")
    col3.metric("Combined", f"{(faith_score+relev_score)/2:.0%}" if run_eval else "Skipped")
    col4.metric("Total Time", f"{total_time:.2f}s")

    if run_eval:
        score_color = color_for_score((faith_score + relev_score) / 2)
        st.markdown(
            f"<div class='metric-box'><b>Combined quality signal:</b> <span style='color:{score_color};font-weight:700;'>{(faith_score + relev_score)/2:.1%}</span></div>",
            unsafe_allow_html=True,
        )

    with st.expander("Retrieved Context Chunks"):
        for i, chunk in enumerate(chunks):
            score_bits = []
            if "rerank_score" in chunk:
                score_bits.append(f"rerank={chunk['rerank_score']:.4f}")
            if "rrf_score" in chunk:
                score_bits.append(f"rrf={chunk['rrf_score']:.5f}")
            if "score" in chunk:
                score_bits.append(f"score={chunk['score']:.4f}")
            score_text = " | ".join(score_bits)

            header = f"Chunk {i+1}: {chunk.get('title', 'Untitled')}"
            if score_text:
                header += f" ({score_text})"

            st.markdown(f"**{header}**")
            st.write(chunk["text"])
            st.divider()

    if run_eval:
        with st.expander("Faithfulness Details"):
            if claims:
                for item in claims:
                    icon = "SUPPORTED" if item["verdict"] == "SUPPORTED" else "UNSUPPORTED"
                    st.write(f"[{icon}] {item['claim']}")
            else:
                st.write("No claims extracted from the answer.")

        with st.expander("Relevancy Details"):
            if questions:
                for i, generated_q in enumerate(questions):
                    sim = relev_sims[i] if i < len(relev_sims) else None
                    if sim is None:
                        st.write(f"{i + 1}. {generated_q}")
                    else:
                        st.write(f"{i + 1}. {generated_q} (similarity={sim:.3f})")
            else:
                st.write("No generated alternate questions were produced by the judge model.")

    with st.expander("Latency Breakdown"):
        st.write(f"Retrieval: {retrieval_sec:.3f}s")
        st.write(f"Generation: {generation_sec:.3f}s")
        if run_eval:
            st.write(f"Faithfulness evaluation: {faith_eval_sec:.3f}s")
            st.write(f"Relevancy evaluation: {relev_eval_sec:.3f}s")

    with st.expander("Debug: Retrieval Pipeline"):
        st.json(ret_result)