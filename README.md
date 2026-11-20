# Semantic Atlas — Text Embedding Explorer

A standalone Streamlit app to explore semantic relationships between any set of texts.

## Features
- **Two embedding backends**: neural (`sentence-transformers`, e.g. MiniLM / mpnet / multilingual) or a fast offline TF-IDF+SVD fallback — no internet/torch required for the fallback.
- **Similarity heatmap** (Plotly, interactive) plus automatic "most similar pair" callout.
- **2D/3D embedding space** with PCA, t-SNE, or UMAP, colored by cluster, hoverable text.
- **Semantic search** — type a query, get ranked nearest texts with a similarity bar chart.
- **Auto-clustering** (KMeans) with automatic k selection via silhouette score, plus per-cluster top TF-IDF keywords.
- **Export** results as CSV (clusters, embeddings) or a full JSON bundle (texts, clusters, similarity matrix).
- Paste text directly or upload a `.txt` / `.csv` file.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

`sentence-transformers` and `umap-learn` are optional — the app still works without them
(it falls back to TF-IDF embeddings and PCA/t-SNE), just with lower embedding quality.
If you skip them, remove those two lines from `requirements.txt` before installing.

## Run

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## Notes
- First run with a Sentence Transformer model will download the model weights (a few hundred MB) — needs an internet connection once, then it's cached.
- For large text sets (500+), UMAP is faster and often gives cleaner clusters than t-SNE.
- On Windows, if `pip install` fails building `numpy`/`scipy` from source, use a supported Python version (3.11–3.12 is safest) or a prebuilt wheel via `pip install numpy scipy --only-binary=:all:`.
