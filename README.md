# Medical RAG Assignment

This repository contains a medical-domain Retrieval-Augmented Generation (RAG) project with:

- Hybrid retrieval (BM25 + semantic retrieval)
- Reciprocal Rank Fusion (RRF)
- Cross-encoder re-ranking
- LLM-as-a-Judge evaluation (faithfulness + relevancy)
- Streamlit web app
- Ablation study across chunking and retrieval configurations

## Project Files

- `scraper.py` - Collect corpus from PubMed
- `chunk_and_index.py` - Chunk corpus, embed, and upsert to Pinecone
- `retriever.py` - Semantic + hybrid retrieval logic
- `generator.py` - Answer generation with HF API and local fallback
- `evaluator.py` - Faithfulness and relevancy evaluator
- `run_evaluation.py` - Ablation runner
- `app.py` - Streamlit UI
- `requirements.txt` - Python dependencies

## Quick Start

1. Create and activate virtual environment
2. Install dependencies
3. Add API keys in `.env`
4. Run app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8503
```

5. Run evaluation:

```powershell
.\.venv\Scripts\python.exe run_evaluation.py --continue-on-error
```

## Included Artifacts

- `data/corpus_clean.json` (cleaned corpus)
- `data/evaluation_results/ablation_summary.json` (ablation output)
