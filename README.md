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

## Deployment

### Streamlit Cloud (Recommended)

1. Ensure code is pushed to GitHub: https://github.com/irfanzainab04/Rag-Assignment
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app" and select this repository
4. Specify `main` branch and `app.py` as the main file
5. Go to **App settings** → **Secrets** and add:
   ```toml
   PINECONE_API_KEY = "your_pinecone_api_key"
   HF_API_TOKEN = "your_huggingface_api_token"
   ```
6. Deploy!

### Docker (Self-hosted)

```bash
docker build -t medical-rag .
docker run -p 8501:8501 \
  -e PINECONE_API_KEY="your_key" \
  -e HF_API_TOKEN="your_token" \
  medical-rag
```

## Included Artifacts

- `data/corpus_clean.json` (cleaned corpus)
- `data/evaluation_results/ablation_summary.json` (ablation output)
