# 🧭 Semantic Atlas
### An Interactive Platform for Semantic Analysis and Meaning Preservation

<p align="center">
  <img src="docs/screenshots/01-home.png" alt="Semantic Atlas Home Page" width="100%">
</p>

<p align="center">
  ![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
  ![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red?logo=streamlit)
  ![NLP](https://img.shields.io/badge/Domain-NLP-green)
  ![Machine Learning](https://img.shields.io/badge/ML-Scikit--learn-orange)
</p>

## 📌 Overview

**Semantic Atlas** is an interactive NLP-based platform that transforms text into high-dimensional embeddings to explore semantic relationships, similarity, clustering, and meaning preservation.

It provides tools to analyze how meaning changes across translation and summarization operations through machine learning, mathematical analysis, and interactive visualizations.

**Project Type:** B.Tech Major Project  
**Institution:** Amity School of Engineering and Technology, Amity University Punjab.

## 🚀 Key Features

The application consists of 11 interactive modules:

- **Overview:** Text analysis, cluster assignments, and outlier detection.
- **Similarity:** Cosine similarity heatmaps and duplicate detection.
- **Embedding Space:** Interactive 2D/3D semantic visualizations.
- **Semantic Search:** Context-based text retrieval.
- **Analogy Lab:** Vector arithmetic and semantic interpolation.
- **Compass:** Analyze texts along custom semantic dimensions.
- **Robustness:** Measure meaning preservation through round-trip translation.
- **Functorial Check:** Compare different sequences of translation and summarization.
- **Clusters:** Semantic grouping, keywords, and cluster insights.
- **Insights:** Zero-shot tagging and coverage gap analysis.
- **Export:** Download embeddings, clusters, JSON data, and HTML reports.

## 🧪 Original Research Contributions

### 1. Semantic Robustness Score

Measures semantic preservation by translating text into another language and back, then comparing the original and reconstructed embeddings using cosine similarity.

\[
SRS = 100 \times \operatorname{mean}(\text{Cosine Similarity})
\]

Results can be compared across texts, languages, and embedding models.

### 2. Functorial Commutativity Check

Evaluates whether changing the order of translation and summarization produces semantically similar results.

- **Path A:** Translate → Summarize → Translate Back
- **Path B:** Summarize → Translate → Translate Back

The final outputs are compared using cosine similarity.

### 3. Coverage Gap Analysis

Identifies underrepresented regions in the embedding space to highlight potential gaps in a text dataset.

## 🏗️ System Architecture

```mermaid
flowchart TD
    A[Text Input / TXT / CSV] --> B[Embedding Layer]
    B --> C[Sentence Transformers]
    B --> D[TF-IDF + SVD]
    C --> E[Semantic Analysis Engine]
    D --> E
    E --> F[Similarity & Clustering]
    E --> G[Search & Visualization]
    E --> H[Coverage Analysis]
    I[Translation Layer] --> J[Research Modules]
    K[Claude API - Optional] --> J
    E --> J
    J --> L[Streamlit Dashboard]
```

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python 3.10+ |
| Framework | Streamlit |
| NLP & Embeddings | Sentence-Transformers |
| Embedding Models | MiniLM, MPNet, Multilingual MiniLM |
| Machine Learning | Scikit-learn |
| Clustering | K-Means, Silhouette Score |
| Dimensionality Reduction | PCA, UMAP, t-SNE |
| Visualization | Plotly |
| Data Processing | NumPy, Pandas |
| Translation | deep-translator |
| LLM Integration | Anthropic Claude API |

## 📂 Project Structure

```text
semantic-atlas/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
└── docs/
    ├── screenshots/
    │   ├── 01-home.png
    │   ├── 02-overview.png
    │   ├── 04-embedding-space.png
    │   └── 08-robustness.png
    ├── Semantic_Atlas_Project_Report.docx
    └── paper.pdf
```

## 💻 Installation & Setup

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/semantic-atlas.git
cd semantic-atlas
```

**2. Create a virtual environment**

```bash
python -m venv .venv
```

**3. Activate the environment**

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

**4. Install dependencies**

```bash
pip install -r requirements.txt
```

**5. Run the application**

```bash
python -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

**Note:** The first run downloads the selected embedding model. Translation and optional AI-powered features require internet connectivity.

## 🖼️ Screenshots

### Home Page
![Home](docs/screenshots/01-home.png)

### Overview Dashboard
![Overview](docs/screenshots/02-overview.png)

### Semantic Embedding Space
![Embedding Space](docs/screenshots/04-embedding-space.png)

### Semantic Robustness Analysis
![Robustness](docs/screenshots/08-robustness.png)

## 🔮 Future Scope

- Standardized multilingual evaluation benchmarks.
- Additional semantic preservation metrics.
- Extended transformation analysis.
- Improved semantic coverage detection.
- Evaluation of additional multilingual embedding models.

## 👥 Authors

- **Sehaj Kaur** – Amity School of Engineering and Technology
- **Mansi** – Amity School of Engineering and Technology

**Project Supervisor:** Dr. Vishwanath Bijalwan

---

<p align="center">
  <strong>Semantic Atlas — Making the Hidden Structure of Meaning Visible.</strong>
</p>

<p align="center">
  Developed with Python, NLP, Machine Learning, and Streamlit.
</p>
