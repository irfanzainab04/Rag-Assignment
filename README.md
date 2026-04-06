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

## Local Replication (Exact Steps)

1. Clone the repository:

```bash
git clone https://github.com/irfanzainab04/Rag-Assignment
cd Rag-Assignment
```

2. Create and activate a Python 3.11 virtual environment:

Linux/macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Add API keys in `.env`:

```dotenv
PINECONE_API_KEY=your_pinecone_api_key
HF_API_TOKEN=your_huggingface_api_token
```

5. Build chunk indexes and BM25 caches:

```bash
python chunk_and_index.py
```

6. Run the Streamlit app:

```bash
streamlit run app.py
```

7. Run ablation evaluation (optional):

```bash
python run_evaluation.py --continue-on-error
```

8. Expected outputs:
- App opens locally and returns answers with retrieved chunks.
- LLM-as-a-Judge scores are visible in UI (Faithfulness and Relevancy).
- Evaluation summary is written to `data/evaluation_results/ablation_summary.json`.

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
