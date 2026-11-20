"""
Semantic Atlas — Text Embedding Explorer

A standalone Streamlit application for analyzing semantic relationships
between texts: similarity, geometry, search, clustering, near-duplicate
detection, diverse-subset sampling, analogy solving, semantic interpolation,
explainable search, an interactive force-directed graph, and optional
AI-generated cluster insights.
"""

import io
import json
import html as html_lib
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from deep_translator import GoogleTranslator
    HAS_TRANSLATOR = True
except ImportError:
    HAS_TRANSLATOR = False


# ----------------------------------------------------------------------------
# Design tokens
# ----------------------------------------------------------------------------
BG = "#0A0E17"
PANEL = "#121826"
ACCENT = "#7C5CFC"
ACCENT_2 = "#22D3EE"
TEXT = "#E5E9F2"
MUTED = "#8B93A7"

PALETTE = ["#7C5CFC", "#22D3EE", "#F472B6", "#34D399", "#FBBF24",
           "#60A5FA", "#F87171", "#A78BFA", "#2DD4BF", "#FB923C"]

# LANGUAGE_OPTIONS = {
#     "English": "en", "Hindi": "hi", "Punjabi": "pa", "French": "fr", "German": "de",
#     "Spanish": "es", "Chinese (Simplified)": "zh-CN", "Japanese": "ja", "Arabic": "ar",
#     "Russian": "ru", "Portuguese": "pt", "Korean": "ko", "Italian": "it",
# }

LANGUAGE_OPTIONS = {
    # Already supported
    "English": "en", "Hindi": "hi", "Punjabi": "pa", "French": "fr", "German": "de",
    "Spanish": "es", "Chinese (Simplified)": "zh-CN", "Japanese": "ja", "Arabic": "ar",
    "Russian": "ru", "Portuguese": "pt", "Korean": "ko", "Italian": "it",
    # Indian languages
    "Bengali": "bn", "Tamil": "ta", "Telugu": "te", "Marathi": "mr", "Gujarati": "gu",
    "Kannada": "kn", "Malayalam": "ml", "Urdu": "ur", "Nepali": "ne",
    # European
    "Dutch": "nl", "Polish": "pl", "Swedish": "sv", "Greek": "el", "Czech": "cs",
    "Romanian": "ro", "Hungarian": "hu", "Ukrainian": "uk", "Turkish": "tr",
    # Asian / other
    "Chinese (Traditional)": "zh-TW", "Indonesian": "id", "Vietnamese": "vi", "Thai": "th",
    "Persian": "fa", "Hebrew": "iw", "Swahili": "sw",
}


def render_welcome():
    st.markdown(
        "<p class='welcome-lead'>Semantic Atlas turns any collection of text — reviews, notes, "
        "support tickets, articles, survey responses — into a navigable map of meaning. It converts "
        "each text into a numeric vector that captures what it's <em>about</em>, then gives you six "
        "different lenses for exploring the relationships between them: which texts are similar, how "
        "they cluster into topics, how they're spread out geometrically, and how they respond to a "
        "search query or a conceptual analogy.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='welcome-heading'>How it works</div>", unsafe_allow_html=True)
    steps = [
        ("1", "Add your texts", "Paste one text per line in the sidebar, or upload a .txt / .csv file. "
                                  "A sample dataset is already loaded so you can try it immediately."),
        ("2", "Choose a model", "Sentence Transformer gives real semantic understanding (recommended). "
                                  "TF-IDF is a lighter, fully offline fallback based on shared vocabulary."),
        ("3", "Click Analyze", "Embeddings, dimensionality reduction, and clustering all run in a single pass — "
                                 "typically a few seconds for small sets."),
        ("4", "Explore the tabs", "Similarity, Embedding Space, Semantic Search, Analogy Lab, Clusters, and "
                                    "Export each surface something different about your data."),
    ]
    scol1, scol2 = st.columns(2)
    for i, (num, title, desc) in enumerate(steps):
        target = scol1 if i % 2 == 0 else scol2
        target.markdown(
            f"<div class='step-row'><div class='step-num'>{num}</div>"
            f"<div class='step-text'><h4>{title}</h4><p>{desc}</p></div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='welcome-heading'>What each tab does</div>", unsafe_allow_html=True)
    features = [
        ("Similarity", "See how closely every text relates to every other text as a heatmap, "
                        "plus automatic near-duplicate detection."),
        ("Embedding Space", "Visualize your texts as points or a live draggable graph in 2D/3D, "
                             "colored by topic cluster."),
        ("Semantic Search", "Search your own texts by meaning rather than keywords, with optional "
                             "word-level explainability."),
        ("Analogy Lab", "Solve sentence-level analogies (A − B + C) and walk a conceptual path "
                         "between two ideas."),
        ("Clusters", "Automatic topic grouping with representative examples, keywords, and "
                      "optional AI-written summaries."),
        ("Export", "Download cluster assignments, raw embeddings, or a full JSON bundle for "
                    "further analysis elsewhere."),
    ]
    cols = st.columns(3)
    for i, (title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(
                f"<div class='glass-card'><h4>{title}</h4><p>{desc}</p></div>",
                unsafe_allow_html=True,
            )
            st.write("")

    st.markdown(
        "<div class='cta-banner'>Ready when you are — the sample dataset in the sidebar is preloaded. "
        "Click <b>Analyze</b> to see the whole app in action, then replace it with your own texts.</div>",
        unsafe_allow_html=True,
    )


def style_fig(fig, height=560, show_axes=True):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Inter, -apple-system, sans-serif"),
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    if not show_axes:
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
    return fig


# ----------------------------------------------------------------------------
# Page config and styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Semantic Atlas — Text Embedding Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }}
    .main > div {{ padding-top: 1rem; }}

    /* scrollbar */
    ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    ::-webkit-scrollbar-track {{ background: {BG}; }}
    ::-webkit-scrollbar-thumb {{ background: rgba(124,92,252,0.35); border-radius: 8px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: rgba(124,92,252,0.6); }}

    /* header */
    .app-header {{
        position: relative;
        padding: 2rem 2.2rem;
        border-radius: 20px;
        margin-bottom: 1.6rem;
        background:
            radial-gradient(circle at 15% 20%, rgba(124,92,252,0.22), transparent 45%),
            radial-gradient(circle at 85% 80%, rgba(34,211,238,0.16), transparent 45%),
            {PANEL};
        border: 1px solid rgba(124,92,252,0.25);
        overflow: hidden;
    }}
    .app-header::before {{
        content: "";
        position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, {ACCENT}, {ACCENT_2}, {ACCENT});
        background-size: 200% auto;
    }}
    .app-header .eyebrow {{
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase;
        color: {ACCENT_2}; margin-bottom: 0.5rem;
    }}
    .app-header h1 {{
        margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -0.02em;
        background: linear-gradient(90deg, #FFFFFF 20%, {ACCENT_2} 100%);
        -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .app-header p {{ margin: 0.5rem 0 0 0; font-size: 0.98rem; color: {MUTED}; max-width: 640px; }}

    /* metric cards */
    div[data-testid="stMetric"] {{
        background: rgba(124,92,252,0.06);
        border: 1px solid rgba(124,92,252,0.22);
        border-radius: 14px;
        padding: 0.9rem 1.1rem 0.7rem 1.1rem;
        transition: border-color 0.2s ease;
    }}
    div[data-testid="stMetric"]:hover {{ border-color: rgba(124,92,252,0.55); }}
    div[data-testid="stMetricValue"] {{
        font-size: 1.65rem; font-weight: 800;
        background: linear-gradient(90deg, {TEXT}, {ACCENT_2});
        -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
    }}
    div[data-testid="stMetricLabel"] {{ color: {MUTED}; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }}

    /* tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px; border-bottom: 1px solid rgba(124,92,252,0.18); background: transparent;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px 8px 0 0; padding: 10px 18px; font-weight: 600;
        color: {MUTED}; background: transparent;
    }}
    .stTabs [aria-selected="true"] {{
        color: {TEXT} !important;
        background: rgba(124,92,252,0.10) !important;
        border-bottom: 2px solid {ACCENT} !important;
    }}

    /* buttons */
    .stButton > button {{
        background: linear-gradient(120deg, {ACCENT}, #5B3FD6);
        color: white; border: none; border-radius: 10px; font-weight: 600;
        box-shadow: 0 4px 14px rgba(124,92,252,0.25);
        transition: transform 0.12s ease, box-shadow 0.12s ease;
    }}
    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(124,92,252,0.4);
    }}
    .stDownloadButton > button {{
        background: {PANEL}; color: {TEXT}; border: 1px solid rgba(124,92,252,0.35);
        border-radius: 10px; font-weight: 600;
    }}
    .stDownloadButton > button:hover {{ border-color: {ACCENT}; color: {ACCENT_2}; }}

    /* sidebar section headers */
    section[data-testid="stSidebar"] h3 {{
        font-size: 0.85rem !important; text-transform: uppercase; letter-spacing: 0.06em;
        color: {ACCENT_2}; border-left: 3px solid {ACCENT}; padding-left: 0.6rem; margin-top: 0.4rem;
    }}
    section[data-testid="stSidebar"] {{ background: {PANEL}; }}

    /* misc */
    .tag {{
        display: inline-block; padding: 3px 11px; border-radius: 999px;
        background: rgba(34,211,238,0.12); color: {ACCENT_2}; font-size: 0.75rem;
        font-weight: 700; margin-right: 6px; border: 1px solid rgba(34,211,238,0.3);
        font-family: 'JetBrains Mono', monospace;
    }}
    .section-caption {{ font-size: 0.85rem; color: {MUTED}; margin-bottom: 0.7rem; line-height: 1.5; }}
    hr {{ margin: 1.2rem 0; border-color: rgba(124,92,252,0.18); }}

    .stExpander {{
        border: 1px solid rgba(124,92,252,0.18) !important; border-radius: 12px !important;
        background: rgba(124,92,252,0.03);
    }}

    .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {{
        background: {BG} !important; border: 1px solid rgba(124,92,252,0.25) !important;
        border-radius: 8px !important; color: {TEXT} !important;
    }}
    .stTextInput input:focus, .stTextArea textarea:focus {{ border-color: {ACCENT} !important; }}

    /* welcome screen */
    .glass-card {{
        background: rgba(124,92,252,0.05);
        border: 1px solid rgba(124,92,252,0.20);
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        height: 100%;
        transition: border-color 0.2s ease, transform 0.15s ease;
    }}
    .glass-card:hover {{ border-color: rgba(124,92,252,0.5); transform: translateY(-2px); }}
    .glass-card h4 {{ margin: 0 0 0.4rem 0; color: {ACCENT_2}; font-size: 1rem; font-weight: 700; }}
    .glass-card p {{ margin: 0; color: {MUTED}; font-size: 0.87rem; line-height: 1.5; }}

    .step-row {{ display: flex; align-items: flex-start; gap: 0.9rem; margin-bottom: 1.1rem; }}
    .step-num {{
        flex-shrink: 0; width: 32px; height: 32px; border-radius: 50%;
        background: linear-gradient(135deg, {ACCENT}, {ACCENT_2});
        color: white; font-weight: 800; font-size: 0.9rem;
        display: flex; align-items: center; justify-content: center;
    }}
    .step-text h4 {{ margin: 0 0 0.15rem 0; color: {TEXT}; font-size: 0.98rem; font-weight: 700; }}
    .step-text p {{ margin: 0; color: {MUTED}; font-size: 0.87rem; line-height: 1.5; }}

    .welcome-heading {{
        font-size: 1.1rem; font-weight: 700; color: {TEXT}; margin: 1.6rem 0 1rem 0;
        letter-spacing: -0.01em;
    }}
    .welcome-lead {{ color: {MUTED}; font-size: 0.97rem; line-height: 1.65; max-width: 780px; }}
    .cta-banner {{
        margin-top: 1.8rem; padding: 1rem 1.2rem; border-radius: 12px;
        background: linear-gradient(120deg, rgba(124,92,252,0.14), rgba(34,211,238,0.10));
        border: 1px solid rgba(124,92,252,0.3);
        color: {TEXT}; font-size: 0.92rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <div class="eyebrow">Embedding Intelligence</div>
        <h1>Semantic Atlas</h1>
        <p>Map meaning across any set of texts — similarity, geometry, search, clustering,
        analogy, and AI-assisted insight, all from one interface.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Caching and core functions
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_st_model(model_name: str):
    return SentenceTransformer(model_name)


@st.cache_data(show_spinner=False)
def embed_texts(texts: tuple, method: str, model_name: str = ""):
    if method == "Sentence Transformer" and HAS_ST:
        model = load_st_model(model_name)
        vecs = model.encode(list(texts), show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(vecs)
    else:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        tfidf = vectorizer.fit_transform(texts)
        n_comp = min(100, max(2, tfidf.shape[1] - 1), tfidf.shape[0] - 1)
        n_comp = max(n_comp, 2)
        svd = TruncatedSVD(n_components=min(n_comp, tfidf.shape[1] - 1) or 2, random_state=42)
        vecs = svd.fit_transform(tfidf)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vecs / norms


# def embed_single_neural(text: str, model_name: str) -> np.ndarray:
#     model = load_st_model(model_name)
#     return np.asarray(model.encode([text], normalize_embeddings=True)[0])

@st.cache_data(show_spinner=False)
def embed_single_neural(text: str, model_name: str) -> np.ndarray:
    model = load_st_model(model_name)
    return np.asarray(model.encode([text], normalize_embeddings=True)[0])


def embed_query(query, texts, embeddings, method, model_name):
    """Returns (query_vec, doc_vecs) in the same vector space."""
    if method == "Sentence Transformer" and HAS_ST:
        return embed_single_neural(query, model_name), embeddings
    combo = embed_texts(tuple(texts) + (query,), method, model_name)
    return combo[-1], combo[:-1]


def embed_words_neural(words: list, model_name: str) -> np.ndarray:
    model = load_st_model(model_name)
    return np.asarray(model.encode(words, normalize_embeddings=True))


@st.cache_data(show_spinner=False)
def round_trip_translate(text: str, target_code: str, source_code: str = "en"):
    """Translates text -> target language -> back to source language. Free, no API key required."""
    forward = GoogleTranslator(source=source_code, target=target_code).translate(text)
    if not forward:
        raise ValueError("empty forward translation")
    back = GoogleTranslator(source=target_code, target=source_code).translate(forward)
    if not back:
        raise ValueError("empty back-translation")
    return forward, back


@st.cache_data(show_spinner=False)
def translate_text(text: str, source_code: str, target_code: str) -> str:
    """One-way translation, cached. Free, no API key required."""
    if source_code == target_code:
        return text
    result = GoogleTranslator(source=source_code, target=target_code).translate(text)
    if not result:
        raise ValueError("empty translation")
    return result


def generate_summary(api_key: str, model_name: str, text: str, language_name: str) -> str:
    """Asks Claude for a one-sentence summary, written in the specified language."""
    client = anthropic.Anthropic(api_key=api_key)
    prompt = (
        f"Summarize the following text in one concise sentence, written in {language_name}. "
        "Respond with only the summary sentence, nothing else.\n\n"
        f"Text: {text}"
    )
    resp = client.messages.create(model=model_name, max_tokens=200,
                                   messages=[{"role": "user", "content": prompt}])
    return "".join(block.text for block in resp.content if hasattr(block, "text")).strip()


def functorial_commutativity_check(api_key, ai_model_name, text, source_code, source_name,
                                    target_code, target_name, embed_model_name):
    """
    Tests whether translate-then-summarize and summarize-then-translate commute, in the
    category-theory sense: does the order of the two operations change the resulting meaning?
    Both paths are brought back to the source language so they can be embedded and compared
    with the same model.
    """
    # Path A: translate -> summarize (in target language) -> translate back
    translated = translate_text(text, source_code, target_code)
    summary_in_target = generate_summary(api_key, ai_model_name, translated, target_name)
    path_a_result = translate_text(summary_in_target, target_code, source_code)

    # Path B: summarize (in source language) -> translate -> translate back
    summary_in_source = generate_summary(api_key, ai_model_name, text, source_name)
    path_b_intermediate = translate_text(summary_in_source, source_code, target_code)
    path_b_result = translate_text(path_b_intermediate, target_code, source_code)

    vec_a = embed_single_neural(path_a_result, embed_model_name)
    vec_b = embed_single_neural(path_b_result, embed_model_name)
    score = float(cosine_similarity([vec_a], [vec_b])[0, 0])

    return {
        "translated": translated, "summary_in_target": summary_in_target, "path_a_result": path_a_result,
        "summary_in_source": summary_in_source, "path_b_intermediate": path_b_intermediate,
        "path_b_result": path_b_result, "score": score,
    }


def fit_reducer(embeddings: np.ndarray, method: str, n_components: int, perplexity: int, seed: int = 42):
    """Returns (coords, reducer). reducer is None for t-SNE (no out-of-sample transform)."""
    n = embeddings.shape[0]
    if method == "UMAP" and HAS_UMAP:
        n_neighbors = max(2, min(15, n - 1))
        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=seed)
        coords = reducer.fit_transform(embeddings)
        return coords, reducer
    elif method == "t-SNE":
        perp = max(2, min(perplexity, n - 1))
        reducer = TSNE(n_components=n_components, perplexity=perp, random_state=seed, init="pca")
        coords = reducer.fit_transform(embeddings)
        return coords, None
    else:
        reducer = PCA(n_components=min(n_components, n, embeddings.shape[1]), random_state=seed)
        coords = reducer.fit_transform(embeddings)
        return coords, reducer


def top_terms_per_cluster(texts, labels, n_terms=6):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
    tfidf = vectorizer.fit_transform(texts)
    terms = np.array(vectorizer.get_feature_names_out())
    result = {}
    for c in sorted(set(labels)):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        mean_scores = np.asarray(tfidf[idx].mean(axis=0)).ravel()
        top_idx = mean_scores.argsort()[::-1][:n_terms]
        result[c] = [terms[i] for i in top_idx if mean_scores[i] > 0]
    return result


def cluster_medoids(embeddings, labels):
    medoids = {}
    for c in sorted(set(labels)):
        idx = np.where(labels == c)[0]
        centroid = embeddings[idx].mean(axis=0, keepdims=True)
        sims = cosine_similarity(centroid, embeddings[idx])[0]
        medoids[c] = idx[np.argmax(sims)]
    return medoids


def mmr_diverse_subset(embeddings, k, sim_matrix=None):
    n = embeddings.shape[0]
    k = min(k, n)
    if sim_matrix is None:
        sim_matrix = cosine_similarity(embeddings)
    selected = [int(np.argmax(sim_matrix.sum(axis=1)))]
    candidates = [i for i in range(n) if i not in selected]
    while len(selected) < k and candidates:
        scores = [sim_matrix[c, selected].max() for c in candidates]
        next_idx = candidates[int(np.argmin(scores))]
        selected.append(next_idx)
        candidates.remove(next_idx)
    return selected


def find_outliers(sim_matrix, texts, top_n=3):
    n = sim_matrix.shape[0]
    if n <= 1:
        return []
    avg_sim = (sim_matrix.sum(axis=1) - 1) / (n - 1)
    order = np.argsort(avg_sim)[:top_n]
    return [(texts[i], float(avg_sim[i])) for i in order]


def solve_analogy(text_a, text_b, text_c, texts, embeddings, embed_method, model_name):
    if embed_method == "Sentence Transformer" and HAS_ST:
        va = embed_single_neural(text_a, model_name)
        vb = embed_single_neural(text_b, model_name)
        vc = embed_single_neural(text_c, model_name)
    else:
        combo = embed_texts(tuple(texts) + (text_a, text_b, text_c), embed_method, model_name)
        va, vb, vc = combo[-3], combo[-2], combo[-1]
    result_vec = va - vb + vc
    norm = np.linalg.norm(result_vec)
    if norm > 0:
        result_vec = result_vec / norm
    sims = cosine_similarity([result_vec], embeddings)[0]
    order = np.argsort(sims)[::-1]
    return order, sims


def interpolate_path(idx_a, idx_b, embeddings, steps=5):
    va, vb = embeddings[idx_a], embeddings[idx_b]
    path = []
    for t in np.linspace(0, 1, steps):
        interp = (1 - t) * va + t * vb
        n = np.linalg.norm(interp)
        if n > 0:
            interp = interp / n
        sims = cosine_similarity([interp], embeddings)[0]
        path.append(int(np.argmax(sims)))
    return path


def highlight_text_by_relevance(text, ref_vec, model_name):
    words = text.split()
    if len(words) < 2:
        return text
    word_vecs = embed_words_neural(words, model_name)
    sims = cosine_similarity([ref_vec], word_vecs)[0]
    lo, hi = sims.min(), sims.max()
    spread = hi - lo if hi > lo else 1.0
    spans = []
    for w, s in zip(words, sims):
        norm = (s - lo) / spread
        alpha = 0.12 + 0.68 * norm
        spans.append(
            f"<span style='background: rgba(124,92,252,{alpha:.2f}); padding:1px 4px; "
            # f"border-radius:4px; margin-right:2px;'>{w}</span>"
            f"border-radius:4px; margin-right:2px;'>{html_lib.escape(w)}</span>"
        )
    return " ".join(spans)


def build_force_graph_html(texts, labels, sim_matrix, threshold, palette):
    nodes = []
    for i, (t, c) in enumerate(zip(texts, labels)):
        color = palette[int(c) % len(palette)]
        short = (t[:30] + "...") if len(t) > 30 else t
        nodes.append({"id": i, "label": short, "title": t, "color": color})

    edges = []
    n = len(texts)
    for a in range(n):
        for b in range(a + 1, n):
            if sim_matrix[a, b] >= threshold:
                edges.append({"from": a, "to": b, "value": float(sim_matrix[a, b])})

    template = """
    <div id="network" style="width:100%; height:600px; background:__BG__;
         border-radius:14px; border:1px solid rgba(124,92,252,0.25);"></div>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <script>
      const nodes = new vis.DataSet(__NODES__);
      const edges = new vis.DataSet(__EDGES__);
      const container = document.getElementById('network');
      const data = { nodes: nodes, edges: edges };
      const options = {
        nodes: { shape: 'dot', size: 14, font: { color: '__TEXT__', size: 12 }, borderWidth: 2 },
        edges: { color: { color: 'rgba(124,92,252,0.35)' }, smooth: false, scaling: { min: 1, max: 4 } },
        physics: {
          stabilization: true,
          barnesHut: { gravitationalConstant: -4000, springLength: 130, springConstant: 0.03 }
        },
        interaction: { hover: true, tooltipDelay: 100, dragNodes: true, zoomView: true }
      };
      new vis.Network(container, data, options);
    </script>
    """
    return (
        template.replace("__NODES__", json.dumps(nodes))
        .replace("__EDGES__", json.dumps(edges))
        .replace("__BG__", BG)
        .replace("__TEXT__", TEXT)
    )


def generate_ai_insights(api_key, model_name, clusters_dict):
    client = anthropic.Anthropic(api_key=api_key)
    payload = {str(c): members[:8] for c, members in clusters_dict.items()}
    prompt = (
        "You are analyzing clusters produced by an embedding-based clustering algorithm on a set of texts. "
        "Respond with strict JSON only, no other text, in exactly this shape:\n"
        '{"clusters": [{"id": <int>, "label": "<3-6 word label>", "summary": "<one sentence>"}], '
        '"overall_theme": "<one sentence describing the dataset as a whole>"}\n\n'
        f"Clusters (id: example member texts): {json.dumps(payload)}"
    )
    resp = client.messages.create(
        model=model_name,
        max_tokens=900,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = "".join(block.text for block in resp.content if hasattr(block, "text")).strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


def compute_axis_vector(pos_text, neg_text, texts, embed_method, model_name):
    """Builds a normalized direction vector from two opposite anchor phrases."""
    if embed_method == "Sentence Transformer" and HAS_ST:
        vpos = embed_single_neural(pos_text, model_name)
        vneg = embed_single_neural(neg_text, model_name)
    else:
        combo = embed_texts(tuple(texts) + (pos_text, neg_text), embed_method, model_name)
        vpos, vneg = combo[-2], combo[-1]
    axis = vpos - vneg
    norm = np.linalg.norm(axis)
    if norm > 0:
        axis = axis / norm
    return axis


def project_on_axis(embeddings, axis_vector):
    """Projects every embedding onto a semantic axis. Values roughly in [-1, 1]."""
    return embeddings @ axis_vector


def cluster_confidence(embeddings, labels):
    """For each text: similarity to its own cluster centroid minus the best-matching other centroid."""
    unique = sorted(set(labels))
    centroid_matrix = np.array([embeddings[np.where(labels == c)[0]].mean(axis=0) for c in unique])
    sims = cosine_similarity(embeddings, centroid_matrix)
    margins = np.zeros(len(labels))
    for i, lab in enumerate(labels):
        own_idx = unique.index(lab)
        own_sim = sims[i, own_idx]
        other_sims = np.delete(sims[i], own_idx)
        best_other = other_sims.max() if len(other_sims) else -1.0
        margins[i] = own_sim - best_other
    return margins


def zero_shot_scores(texts, candidate_labels, embed_method, model_name):
    """Embedding-similarity zero-shot tagging: scores every text against every candidate label."""
    if embed_method == "Sentence Transformer" and HAS_ST:
        text_vecs = embed_texts(tuple(texts), embed_method, model_name)
        label_vecs = embed_words_neural(candidate_labels, model_name)
    else:
        combo = embed_texts(tuple(texts) + tuple(candidate_labels), embed_method, model_name)
        text_vecs, label_vecs = combo[: len(texts)], combo[len(texts):]
    return cosine_similarity(text_vecs, label_vecs)


def generate_cluster_contrast(api_key, model_name, label_a, texts_a, label_b, texts_b):
    """Calls Claude to articulate the core conceptual difference between two clusters."""
    client = anthropic.Anthropic(api_key=api_key)
    prompt = (
        "Compare these two groups of texts, each from a different cluster produced by an embedding-based "
        "clustering algorithm. In 2-3 sentences, describe the single biggest conceptual difference between "
        "them, grounded in what is actually said. Do not just restate keywords — explain the underlying "
        "distinction.\n\n"
        f"Cluster {label_a}: {json.dumps(texts_a[:8])}\n\n"
        f"Cluster {label_b}: {json.dumps(texts_b[:8])}"
    )
    resp = client.messages.create(
        model=model_name, max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in resp.content if hasattr(block, "text")).strip()


def build_offline_report_html(texts, labels, sim_matrix, cluster_terms, medoids,
                               k_final, sil, embed_method_used, coords):
    """Bakes the key results into one static, shareable HTML file — no Streamlit runtime required."""
    labels_short = [t[:40] + ("…" if len(t) > 40 else "") for t in texts]
    fig_heat = px.imshow(
        sim_matrix, x=labels_short, y=labels_short,
        color_continuous_scale=[[0, PANEL], [0.5, "#4C3FA8"], [1, ACCENT_2]], aspect="auto",
    )
    style_fig(fig_heat, height=520)

    plot_df = pd.DataFrame(coords[:, :2], columns=["x", "y"])
    plot_df["Cluster"] = labels.astype(str)
    plot_df["Text"] = [t[:80] for t in texts]
    fig_scatter = px.scatter(plot_df, x="x", y="y", color="Cluster", hover_data=["Text"],
                              color_discrete_sequence=PALETTE)
    style_fig(fig_scatter, height=520)

    heat_html = fig_heat.to_html(full_html=False, include_plotlyjs="cdn")
    scatter_html = fig_scatter.to_html(full_html=False, include_plotlyjs=False)

    cluster_html = ""
    for c in sorted(set(labels)):
        members = [t for t, l in zip(texts, labels) if l == c]
        terms = ", ".join(cluster_terms.get(c, [])) or "N/A"
        # rep = texts[medoids[c]]
        # member_list = "".join(f"<li>{m}</li>" for m in members)
        rep = html_lib.escape(texts[medoids[c]])
        member_list = "".join(f"<li>{html_lib.escape(m)}</li>" for m in members)
        cluster_html += (
            f'<div class="cluster-card"><h3>Cluster {c} — {len(members)} texts</h3>'
            f"<p><b>Representative:</b> {rep}</p><p><b>Keywords:</b> {terms}</p>"
            f"<ul>{member_list}</ul></div>"
        )

    sil_text = f"{sil:.3f}" if not np.isnan(sil) else "N/A"
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Semantic Atlas Report</title>
<style>
body {{ background:{BG}; color:{TEXT}; font-family: Inter, -apple-system, sans-serif; padding:2rem; max-width:1100px; margin:0 auto; }}
h1 {{ background: linear-gradient(90deg, #fff, {ACCENT_2}); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
h2 {{ color:{ACCENT_2}; margin-top:2rem; }}
.meta {{ color:{MUTED}; margin-bottom:2rem; }}
.cluster-card {{ background:{PANEL}; border:1px solid rgba(124,92,252,0.25); border-radius:12px; padding:1.2rem 1.4rem; margin-bottom:1rem; }}
.cluster-card h3 {{ margin-top:0; color:{ACCENT_2}; }}
.cluster-card li {{ margin-bottom:0.2rem; }}
</style></head>
<body>
<h1>Semantic Atlas — Analysis Report</h1>
<p class="meta">{len(texts)} texts · {embed_method_used} embeddings · {k_final} clusters · silhouette {sil_text}</p>
<h2>Similarity heatmap</h2>
{heat_html}
<h2>Embedding space</h2>
{scatter_html}
<h2>Clusters</h2>
{cluster_html}
</body></html>"""


def find_gap_regions(coords_2d, n_candidates=400, n_gaps=5, seed=42):
    """
    Grid-samples candidate points across the bounding box of the plotted embedding space
    and returns the sparsest ones (farthest from any real text) as coverage gaps, spread
    apart from each other via simple non-max suppression.
    """
    rng = np.random.default_rng(seed)
    x_min, x_max = coords_2d[:, 0].min(), coords_2d[:, 0].max()
    y_min, y_max = coords_2d[:, 1].min(), coords_2d[:, 1].max()
    x_pad = (x_max - x_min) * 0.08 or 1.0
    y_pad = (y_max - y_min) * 0.08 or 1.0

    candidates = np.column_stack([
        rng.uniform(x_min - x_pad, x_max + x_pad, n_candidates),
        rng.uniform(y_min - y_pad, y_max + y_pad, n_candidates),
    ])

    dists = np.linalg.norm(candidates[:, None, :] - coords_2d[None, :, :], axis=2)
    nearest_dist = dists.min(axis=1)
    order = np.argsort(nearest_dist)[::-1]

    min_sep = max(x_max - x_min, y_max - y_min) * 0.12 or 0.1
    selected = []
    for idx in order:
        pt = candidates[idx]
        if all(np.linalg.norm(pt - candidates[s]) > min_sep for s in selected):
            selected.append(idx)
        if len(selected) >= n_gaps:
            break
    return candidates[selected], nearest_dist[selected]


def generate_gap_suggestions(api_key, model_name, gap_contexts):
    """Calls Claude to hypothesize what kind of content would fill each coverage gap."""
    client = anthropic.Anthropic(api_key=api_key)
    prompt = (
        "Each item below is a 'gap' in a semantic map: a region of meaning-space with no example text "
        "nearby. For each gap, the nearest existing texts are listed. In one short sentence per gap, "
        "hypothesize what kind of new content would plausibly sit there — a topic or angle the dataset "
        "does not currently cover. Respond with strict JSON only, no other text:\n"
        '{"gaps": [{"id": <int>, "suggestion": "<one sentence>"}]}\n\n'
        f"Gaps: {json.dumps(gap_contexts)}"
    )
    resp = client.messages.create(
        model=model_name, max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = "".join(block.text for block in resp.content if hasattr(block, "text")).strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Setup")

    input_mode = st.radio("Input method", ["Paste text", "Upload file"], horizontal=True)

    default_sample = (
        "The cat sat quietly on the warm windowsill.\n"
        "A kitten napped in the afternoon sun.\n"
        "Stock markets rallied after the interest rate decision.\n"
        "Investors cheered the central bank's policy shift.\n"
        "The chef simmered a rich tomato sauce for hours.\n"
        "Fresh basil and garlic gave the pasta dish its aroma.\n"
        "NASA announced a new mission to study Jupiter's moons.\n"
        "The rover sent back stunning images of the Martian surface."
    )

    texts = []
    if input_mode == "Paste text":
        raw = st.text_area("One text per line", value=default_sample, height=220)
        texts = [t.strip() for t in raw.split("\n") if t.strip()]
    else:
        uploaded = st.file_uploader("Upload .txt (one per line) or .csv", type=["txt", "csv"])
        if uploaded is not None:
            if uploaded.name.endswith(".csv"):
                df_up = pd.read_csv(uploaded)
                col = st.selectbox("Text column", df_up.columns)
                texts = df_up[col].dropna().astype(str).tolist()
            else:
                content = uploaded.read().decode("utf-8", errors="ignore")
                texts = [t.strip() for t in content.split("\n") if t.strip()]

    st.markdown("---")
    st.subheader("Embedding model")

    method_options = []
    if HAS_ST:
        method_options.append("Sentence Transformer")
    method_options.append("TF-IDF (offline)")
    embed_method = st.selectbox("Method", method_options, index=0)

    model_name = "all-MiniLM-L6-v2"
    if embed_method == "Sentence Transformer":
        model_name = st.selectbox(
            "Model",
            ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-multilingual-MiniLM-L12-v2"],
            help="MiniLM: fastest. mpnet: highest quality. multilingual: non-English text.",
        )
    elif not HAS_ST:
        st.caption("Install sentence-transformers for neural embeddings.")

    st.markdown("---")
    st.subheader("Dimensionality reduction")
    reduce_options = ["PCA"]
    if HAS_UMAP:
        reduce_options.insert(0, "UMAP")
    reduce_options.append("t-SNE")
    reduce_method = st.selectbox("Method", reduce_options)
    n_dims = st.radio("Plot dimensions", [2, 3], horizontal=True, index=0)
    perplexity = 30
    if reduce_method == "t-SNE":
        perplexity = st.slider("Perplexity", 2, 50, 15)

    st.markdown("---")
    st.subheader("Clustering")
    auto_k = st.checkbox("Auto-select cluster count", value=True)
    manual_k = st.slider("Number of clusters (k)", 2, 10, 3, disabled=auto_k)

    st.markdown("---")
    with st.expander("AI insights (optional)"):
        st.caption(
            "Uses your own Anthropic API key to generate a natural-language label and "
            "summary for each cluster. Nothing is sent until you click Generate in the Clusters tab."
        )
        ai_key = st.text_input("Anthropic API key", type="password")
        ai_model = st.text_input("Model", value="claude-haiku-4-5-20251001")
        if not HAS_ANTHROPIC:
            st.caption("Install the `anthropic` package to enable this.")

    run_btn = st.button("Analyze", type="primary", use_container_width=True)


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------
if "computed" not in st.session_state:
    st.session_state.computed = False

if run_btn:
    if len(texts) < 3:
        st.warning("Add at least 3 texts to get meaningful results.")
    else:
        with st.spinner("Computing embeddings..."):
            t0 = time.time()
            embeddings = embed_texts(tuple(texts), embed_method, model_name)
            elapsed = time.time() - t0

        with st.spinner("Reducing dimensions..."):
            coords, reducer = fit_reducer(embeddings, reduce_method, n_dims, perplexity)

        with st.spinner("Clustering..."):
            n = len(texts)
            if auto_k:
                best_k, best_score = 2, -1
                max_k = min(8, n - 1)
                for k in range(2, max(3, max_k + 1)):
                    if k >= n:
                        break
                    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(embeddings)
                    if len(set(km.labels_)) < 2:
                        continue
                    score = silhouette_score(embeddings, km.labels_)
                    if score > best_score:
                        best_score, best_k = score, k
                k_final = best_k
            else:
                k_final = min(manual_k, n - 1)

            kmeans = KMeans(n_clusters=k_final, random_state=42, n_init=10).fit(embeddings)
            labels = kmeans.labels_
            try:
                sil = silhouette_score(embeddings, labels)
            except ValueError:
                sil = float("nan")

        history_entry = {
            "time": time.strftime("%H:%M:%S"),
            "texts": len(texts), "method": embed_method, "reduction": reduce_method,
            "k": k_final, "silhouette": round(float(sil), 3) if not np.isnan(sil) else None,
        }
        run_history = (st.session_state.get("run_history", []) + [history_entry])[-5:]

        st.session_state.update(
            texts=texts, embeddings=embeddings, coords=coords, reducer=reducer,
            labels=labels, k_final=k_final, sil=sil, elapsed=elapsed,
            embed_method=embed_method, model_name=model_name, reduce_method=reduce_method,
            n_dims=n_dims, what_if_point=None, ai_insights=None, compass_axes=[],
            gap_points=None, gap_dists=None, robustness_df=None, robustness_targets=None,
            robustness_source=None, multi_model_df=None, functorial_result=None,
            run_history=run_history, computed=True,
        )

if st.session_state.computed:
    texts = st.session_state.texts
    embeddings = st.session_state.embeddings
    coords = st.session_state.coords
    reducer = st.session_state.reducer
    labels = st.session_state.labels
    k_final = st.session_state.k_final
    sil = st.session_state.sil
    n_dims = st.session_state.n_dims
    embed_method_used = st.session_state.embed_method
    model_name_used = st.session_state.model_name

    sim_matrix = cosine_similarity(embeddings)
    cluster_terms = top_terms_per_cluster(texts, labels)
    medoids = cluster_medoids(embeddings, labels)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Texts analyzed", len(texts))
    c2.metric("Embedding dimension", embeddings.shape[1])
    c3.metric("Clusters found", k_final)
    c4.metric("Silhouette score", f"{sil:.3f}" if not np.isnan(sil) else "N/A")

    st.write("")

    (tab_overview, tab_sim, tab_space, tab_search, tab_analogy, tab_compass,
     tab_robust, tab_functorial, tab_clusters, tab_insights, tab_export) = st.tabs(
        ["Overview", "Similarity", "Embedding Space", "Semantic Search", "Analogy Lab",
         "Compass", "Robustness", "Functorial Check", "Clusters", "Insights", "Export"]
    )

    # --- Overview ---
    with tab_overview:
        df_preview = pd.DataFrame({"#": range(1, len(texts) + 1), "Text": texts, "Cluster": labels})
        st.dataframe(df_preview, use_container_width=True, height=380)

        st.markdown("---")
        with st.expander("Outlier detection — texts least similar to the rest of the set"):
            for t, score in find_outliers(sim_matrix, texts, top_n=min(3, len(texts))):
                st.write(f"- {t}  (average similarity {score:.3f})")

        run_history = st.session_state.get("run_history", [])
        if len(run_history) > 1:
            st.markdown("---")
            with st.expander("Run history — compare recent analyses"):
                st.markdown(
                    "<div class='section-caption'>The last few times you clicked Analyze in this session, "
                    "so you can see how changing settings affected the result.</div>",
                    unsafe_allow_html=True,
                )
                st.dataframe(pd.DataFrame(run_history), use_container_width=True, hide_index=True)

    # --- Similarity ---
    with tab_sim:
        labels_short = [t[:40] + ("…" if len(t) > 40 else "") for t in texts]
        fig = px.imshow(
            sim_matrix, x=labels_short, y=labels_short,
            color_continuous_scale=[[0, PANEL], [0.5, "#4C3FA8"], [1, ACCENT_2]],
            aspect="auto", labels=dict(color="Cosine similarity"),
        )
        st.plotly_chart(style_fig(fig, height=560), use_container_width=True)

        tri = sim_matrix.copy()
        np.fill_diagonal(tri, -1)
        i, j = np.unravel_index(np.argmax(tri), tri.shape)
        st.markdown("**Most similar pair**")
        colA, colB = st.columns(2)
        colA.info(texts[i])
        colB.info(texts[j])
        st.caption(f"Similarity: {tri[i, j]:.3f}")

        st.markdown("---")
        st.markdown("**Near-duplicate detection**")
        st.markdown(
            "<div class='section-caption'>Flags text pairs whose meaning is nearly identical — "
            "useful for cleaning a dataset before downstream use.</div>",
            unsafe_allow_html=True,
        )
        dup_threshold = st.slider("Similarity threshold", 0.80, 0.999, 0.93, 0.001)
        pairs = []
        n = len(texts)
        for a in range(n):
            for b in range(a + 1, n):
                if sim_matrix[a, b] >= dup_threshold:
                    pairs.append((texts[a], texts[b], sim_matrix[a, b]))
        if pairs:
            dup_df = pd.DataFrame(pairs, columns=["Text A", "Text B", "Similarity"]).sort_values(
                "Similarity", ascending=False
            )
            st.dataframe(dup_df, use_container_width=True, hide_index=True)
        else:
            st.caption(f"No pairs found at or above {dup_threshold:.3f} similarity.")

    # --- Embedding space ---
    with tab_space:
        view = st.radio(
            "View", ["Scatter plot", "Static network", "Interactive force graph"], horizontal=True
        )

        plot_df = pd.DataFrame(coords[:, :n_dims], columns=["x", "y", "z"][:n_dims])
        plot_df["Cluster"] = labels.astype(str)
        plot_df["Text"] = [t[:80] + ("…" if len(t) > 80 else "") for t in texts]

        if view == "Scatter plot":
            if n_dims == 3:
                fig3d = px.scatter_3d(
                    plot_df, x="x", y="y", z="z", color="Cluster", hover_data=["Text"],
                    color_discrete_sequence=PALETTE,
                )
                fig3d.update_traces(marker=dict(size=6, line=dict(width=0.5, color=BG)))
                st.plotly_chart(style_fig(fig3d, height=620), use_container_width=True)
            else:
                fig2d = px.scatter(
                    plot_df, x="x", y="y", color="Cluster", hover_data=["Text"],
                    color_discrete_sequence=PALETTE,
                )
                fig2d.update_traces(marker=dict(size=12, line=dict(width=1, color=BG)))
                st.plotly_chart(style_fig(fig2d, height=560), use_container_width=True)

        elif view == "Static network":
            st.markdown(
                "<div class='section-caption'>Nodes use the reduced coordinates above; an edge is drawn "
                "when similarity exceeds the threshold.</div>", unsafe_allow_html=True,
            )
            edge_threshold = st.slider("Edge similarity threshold", 0.30, 0.99, 0.60, 0.01, key="static_thr")
            edge_x, edge_y = [], []
            for a in range(len(texts)):
                for b in range(a + 1, len(texts)):
                    if sim_matrix[a, b] >= edge_threshold:
                        edge_x += [coords[a, 0], coords[b, 0], None]
                        edge_y += [coords[a, 1], coords[b, 1], None]
            fig_net = go.Figure()
            fig_net.add_trace(go.Scatter(
                x=edge_x, y=edge_y, mode="lines",
                line=dict(width=0.7, color="rgba(124,92,252,0.35)"), hoverinfo="none",
            ))
            for c in sorted(set(labels)):
                idx = np.where(labels == c)[0]
                fig_net.add_trace(go.Scatter(
                    x=coords[idx, 0], y=coords[idx, 1], mode="markers",
                    marker=dict(size=11, color=PALETTE[c % len(PALETTE)], line=dict(width=1, color=BG)),
                    text=[texts[i][:80] for i in idx], hoverinfo="text", name=f"Cluster {c}",
                ))
            st.plotly_chart(style_fig(fig_net, height=580, show_axes=False), use_container_width=True)

        else:  # Interactive force graph
            st.markdown(
                "<div class='section-caption'>Physics-based graph — drag nodes, scroll to zoom, hover for "
                "full text. Layout is computed live in the browser rather than from fixed coordinates.</div>",
                unsafe_allow_html=True,
            )
            force_threshold = st.slider("Edge similarity threshold", 0.30, 0.99, 0.60, 0.01, key="force_thr")
            html = build_force_graph_html(texts, labels, sim_matrix, force_threshold, PALETTE)
            components.html(html, height=620, scrolling=False)

        st.markdown("---")
        with st.expander("Add a hypothetical text (live projection)"):
            st.markdown(
                "<div class='section-caption'>Embeds a new sentence and places it in the existing space "
                "above without recomputing the rest. Requires the Sentence Transformer method with PCA or "
                "UMAP (t-SNE has no defined transform for new points).</div>",
                unsafe_allow_html=True,
            )
            can_project = (embed_method_used == "Sentence Transformer") and (reducer is not None)
            new_text = st.text_input("New text", key="what_if_input", disabled=not can_project)
            if not can_project:
                st.caption("Not available with the current method/reduction combination.")
            elif st.button("Project", disabled=not new_text):
                new_vec = embed_single_neural(new_text, model_name_used)
                new_coord = reducer.transform(new_vec.reshape(1, -1))[0]
                sims_to_existing = cosine_similarity([new_vec], embeddings)[0]
                nearest_idx = np.argsort(sims_to_existing)[::-1][:3]
                st.session_state.what_if_point = (new_text, new_coord)
                st.write("Nearest existing texts:")
                for i in nearest_idx:
                    st.write(f"- {texts[i]}  ({sims_to_existing[i]:.3f})")

            if st.session_state.get("what_if_point"):
                wtxt, wcoord = st.session_state.what_if_point
                fig_wi = px.scatter(
                    plot_df, x="x", y="y", color="Cluster", hover_data=["Text"],
                    color_discrete_sequence=PALETTE,
                )
                fig_wi.add_trace(go.Scatter(
                    x=[wcoord[0]], y=[wcoord[1]], mode="markers",
                    marker=dict(size=16, color="white", symbol="x", line=dict(width=2, color=ACCENT_2)),
                    name="New text", text=[wtxt], hoverinfo="text",
                ))
                st.plotly_chart(style_fig(fig_wi, height=480), use_container_width=True)

    # # --- Semantic search ---
    # with tab_search:
    #     query = st.text_input("Search query", placeholder="e.g. financial news")
    #     top_k = st.slider("Results to show", 1, min(10, len(texts)), min(5, len(texts)))
    #     if query:
    #         q_vec = embed_texts(tuple(texts) + (query,), embed_method_used, model_name_used)[-1]
    #         sims = cosine_similarity([q_vec], embeddings)[0]
    #         order = np.argsort(sims)[::-1][:top_k]

    #         bar_df = pd.DataFrame({
    #             "Text": [texts[i][:60] + ("…" if len(texts[i]) > 60 else "") for i in order],
    #             "Similarity": sims[order],
    #         })
    #         fig_bar = px.bar(
    #             bar_df, x="Similarity", y="Text", orientation="h", range_x=[0, 1],
    #             color="Similarity", color_continuous_scale=[[0, PANEL], [1, ACCENT]],
    #         )
    #         fig_bar.update_layout(yaxis=dict(autorange="reversed"))
    #         st.plotly_chart(style_fig(fig_bar, height=90 + 45 * len(order)), use_container_width=True)

    #         show_explain = embed_method_used == "Sentence Transformer" and st.checkbox(
    #             "Explain matches (highlight influential words)", value=False
    #         )

    #         for rank, i in enumerate(order, 1):
    #             st.markdown(f"**{rank}.** <span class='tag'>score {sims[i]:.3f}</span>", unsafe_allow_html=True)
    #             if show_explain:
    #                 st.markdown(highlight_text_by_relevance(texts[i], q_vec, model_name_used),
    #                             unsafe_allow_html=True)
    #             else:
    #                 st.write(texts[i])
    #     else:
    #         st.caption("Enter a query to find the most semantically similar texts.")
    

        # --- Semantic search ---
    with tab_search:
        query = st.text_input("Search query", placeholder="e.g. financial news")
        col_k, col_min = st.columns(2)
        top_k = col_k.slider("Max results", 1, min(10, len(texts)), min(5, len(texts)))
        min_score = col_min.slider("Minimum similarity", 0.0, 0.8, 0.20, 0.05)

        if query:
            q_vec, doc_vecs = embed_query(query, texts, embeddings, embed_method_used, model_name_used)
            sims = cosine_similarity([q_vec], doc_vecs)[0]
            order = [i for i in np.argsort(sims)[::-1][:top_k] if sims[i] >= min_score]

            if not order:
                st.info("No texts are semantically close to this query. Try lowering the minimum similarity.")
            else:
                bar_df = pd.DataFrame({
                    "Text": [texts[i][:60] + ("…" if len(texts[i]) > 60 else "") for i in order],
                    "Similarity": sims[order],
                })
                fig_bar = px.bar(
                    bar_df, x="Similarity", y="Text", orientation="h",
                    range_x=[0, max(0.5, float(sims[order].max()) * 1.15)],
                    color="Similarity", color_continuous_scale=[[0, PANEL], [1, ACCENT]],
                )
                fig_bar.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(style_fig(fig_bar, height=90 + 45 * len(order)), use_container_width=True)

                show_explain = embed_method_used == "Sentence Transformer" and st.checkbox(
                    "Explain matches (highlight influential words)", value=False
                )
                for rank, i in enumerate(order, 1):
                    st.markdown(f"**{rank}.** <span class='tag'>score {sims[i]:.3f}</span>",
                                unsafe_allow_html=True)
                    if show_explain:
                        st.markdown(highlight_text_by_relevance(texts[i], q_vec, model_name_used),
                                    unsafe_allow_html=True)
                    else:
                        st.write(texts[i])
        else:
            st.caption("Enter a query to find the most semantically similar texts.")



    # --- Analogy Lab ---
    with tab_analogy:
        st.markdown("**Vector analogy solver**")
        st.markdown(
            "<div class='section-caption'>Computes A − B + C in embedding space and finds the closest "
            "existing text to the result — the sentence-level analogue of \"king − man + woman ≈ queen\".</div>",
            unsafe_allow_html=True,
        )
        colA, colB, colC = st.columns(3)
        ta = colA.text_input("Text A", key="an_a")
        tb = colB.text_input("Text B (subtract)", key="an_b")
        tc = colC.text_input("Text C", key="an_c")
        if st.button("Solve analogy") and ta and tb and tc:
            order, sims = solve_analogy(ta, tb, tc, texts, embeddings, embed_method_used, model_name_used)
            st.write("A − B + C is closest in meaning to:")
            for i in order[:3]:
                st.write(f"- {texts[i]}  ({sims[i]:.3f})")

        st.markdown("---")
        st.markdown("**Semantic interpolation**")
        st.markdown(
            "<div class='section-caption'>Walks from one text to another through embedding space, "
            "showing the closest real text at each step — a conceptual path between two ideas.</div>",
            unsafe_allow_html=True,
        )
        idx_a = st.selectbox("Start text", range(len(texts)), format_func=lambda i: texts[i][:60], key="interp_a")
        idx_b = st.selectbox(
            "End text", range(len(texts)), format_func=lambda i: texts[i][:60],
            index=min(1, len(texts) - 1), key="interp_b",
        )
        steps = st.slider("Steps", 3, 8, 5)
        if st.button("Generate path"):
            if idx_a == idx_b:
                st.caption("Choose two different texts.")
            else:
                path = interpolate_path(idx_a, idx_b, embeddings, steps)
                for i, p in enumerate(path):
                    marker = "Start" if i == 0 else ("End" if i == len(path) - 1 else f"Step {i}")
                    st.write(f"**{marker}:** {texts[p]}")

    # --- Compass ---
    with tab_compass:
        st.markdown("**Embedding compass**")
        st.markdown(
            "<div class='section-caption'>Define a custom semantic axis from two opposite anchor phrases "
            "(e.g. \"formal\" vs \"casual\", or \"optimistic\" vs \"pessimistic\"). Every text is then scored "
            "along that spectrum — turning the embedding space into an interpretable dial instead of just "
            "raw distances.</div>", unsafe_allow_html=True,
        )
        colp, coln = st.columns(2)
        pos_label = colp.text_input("Positive pole", placeholder="e.g. optimistic", key="compass_pos")
        neg_label = coln.text_input("Negative pole", placeholder="e.g. pessimistic", key="compass_neg")
        axis_name = st.text_input("Axis name (optional)", placeholder="e.g. Tone", key="compass_name")

        if "compass_axes" not in st.session_state:
            st.session_state.compass_axes = []

        if st.button("Add axis") and pos_label and neg_label:
            axis_vec = compute_axis_vector(pos_label, neg_label, texts, embed_method_used, model_name_used)
            scores = project_on_axis(embeddings, axis_vec)
            name = axis_name or f"{pos_label} \u2194 {neg_label}"
            st.session_state.compass_axes.append(
                {"name": name, "pos": pos_label, "neg": neg_label, "scores": scores}
            )

        if st.session_state.compass_axes:
            for ax_i, axis in enumerate(st.session_state.compass_axes):
                st.markdown(f"---\n**{axis['name']}**")
                axis_df = pd.DataFrame({
                    "score": axis["scores"], "y": [0] * len(texts),
                    "Text": [t[:70] + ("…" if len(t) > 70 else "") for t in texts],
                    "Cluster": labels.astype(str),
                })
                fig_ax = px.scatter(
                    axis_df, x="score", y="y", color="Cluster", hover_data=["Text"],
                    color_discrete_sequence=PALETTE,
                )
                fig_ax.update_traces(marker=dict(size=13, line=dict(width=1, color=BG)))
                fig_ax.update_yaxes(visible=False, range=[-1, 1])
                fig_ax.update_xaxes(title=f"{axis['neg']}  \u2190  \u2192  {axis['pos']}", range=[-1, 1])
                st.plotly_chart(style_fig(fig_ax, height=220), use_container_width=True)

                order = np.argsort(axis["scores"])
                colL, colR = st.columns(2)
                with colL:
                    st.caption(f'Closest to "{axis["neg"]}"')
                    for i in order[:3]:
                        st.write(f"- {texts[i]}  ({axis['scores'][i]:.3f})")
                with colR:
                    st.caption(f'Closest to "{axis["pos"]}"')
                    for i in order[::-1][:3]:
                        st.write(f"- {texts[i]}  ({axis['scores'][i]:.3f})")

                if st.button("Remove this axis", key=f"remove_axis_{ax_i}"):
                    st.session_state.compass_axes.pop(ax_i)
                    st.rerun()
        else:
            st.caption("No axes defined yet. Add one above to get started.")

    # --- Robustness ---
    with tab_robust:
        st.markdown("**Semantic Robustness Score**")
        st.markdown(
            "<div class='section-caption'>Round-trip translates each text through another language and "
            "back to its original language, then measures how much the embedding drifted from the "
            "original with that model. Higher similarity means the model preserved meaning better "
            "through translation. This uses free machine translation (no API key needed) — only the "
            "embedding step uses your chosen model.</div>",
            unsafe_allow_html=True,
        )

        if not HAS_TRANSLATOR:
            st.caption("Install the `deep-translator` package to enable this feature.")
        elif embed_method_used != "Sentence Transformer":
            st.caption(
                "Requires the Sentence Transformer embedding method — select it in the sidebar and "
                "click Analyze again."
            )
        else:
            source_lang_name = st.selectbox(
                "Source language of your texts", list(LANGUAGE_OPTIONS.keys()), index=0, key="rb_source"
            )
            target_pool = [name for name in LANGUAGE_OPTIONS if name != source_lang_name]
            default_targets = [l for l in ["Hindi", "French", "German"] if l in target_pool][:3] or target_pool[:2]
            lang_choices = st.multiselect(
                "Languages to round-trip through", target_pool, default=default_targets, key="rb_targets"
            )

            if st.button("Run robustness analysis") and lang_choices:
                source_code = LANGUAGE_OPTIONS[source_lang_name]
                target_codes = {name: LANGUAGE_OPTIONS[name] for name in lang_choices}
                total = len(texts) * len(target_codes)
                progress = st.progress(0.0, text="Translating and re-embedding...")
                # rows, done = [], 0
                rows, done, last_error = [], 0, None

                for ti, text in enumerate(texts):
                    for lang_name, code in target_codes.items():
                        try:
                            _, back = round_trip_translate(text, code, source_code)
                            back_vec = embed_single_neural(back, model_name_used)
                            sim = float(cosine_similarity([embeddings[ti]], [back_vec])[0, 0])
                            rows.append({
                                "Text": text, "Language": lang_name, "Similarity": sim,
                                "Round-tripped": back,
                            })
                        # except Exception:
                        #     rows.append({
                        #         "Text": text, "Language": lang_name, "Similarity": None,
                        #         "Round-tripped": None,
                        #     })

                        except Exception as e:
                            last_error = f"{lang_name}: {type(e).__name__}: {e}"
                            rows.append({
                                "Text": text, "Language": lang_name, "Similarity": None,
                                "Round-tripped": None,
                            })

                        done += 1
                        progress.progress(done / total, text=f"Translating and re-embedding... ({done}/{total})")
                progress.empty()
                st.session_state.robustness_df = pd.DataFrame(rows)
                st.session_state.robustness_targets = target_codes
                st.session_state.robustness_source = source_code
                st.session_state.robustness_error = last_error


            if st.session_state.get("robustness_df") is not None:
                full_df = st.session_state.robustness_df
                df = full_df.dropna(subset=["Similarity"])
                if df.empty:
                    st.warning("No successful translations — check your internet connection and try again.")
                    if st.session_state.get("robustness_error"):
                        st.code(st.session_state.robustness_error)
                    # st.warning(
                    #     "No successful translations — check your internet connection and try again."
                    # )
                else:
                    overall_score = df["Similarity"].mean() * 100
                    n_failed = len(full_df) - len(df)

                    c1r, c2r = st.columns(2)
                    c1r.metric("Semantic Robustness Score", f"{overall_score:.1f} / 100")
                    c2r.metric("Round-trips completed", f"{len(df)} / {len(full_df)}")
                    if n_failed:
                        st.caption(f"{n_failed} translation(s) failed and were excluded (network or rate limits).")

                    st.markdown("---")
                    st.markdown("**Robustness by text**")
                    by_text = df.groupby("Text")["Similarity"].mean().sort_values()
                    fig_text = px.bar(
                        x=by_text.values, y=[t[:60] + ("…" if len(t) > 60 else "") for t in by_text.index],
                        orientation="h", color=by_text.values,
                        color_continuous_scale=[[0, "#F87171"], [0.5, PANEL], [1, ACCENT_2]],
                        range_color=[0, 1],
                    )
                    fig_text.update_layout(xaxis_title="Average similarity after round-trip", yaxis_title="")
                    st.plotly_chart(style_fig(fig_text, height=90 + 35 * len(by_text)), use_container_width=True)

                    st.markdown("**Robustness by language**")
                    by_lang = df.groupby("Language")["Similarity"].mean().sort_values()
                    fig_lang = px.bar(
                        x=by_lang.values, y=by_lang.index, orientation="h", color=by_lang.values,
                        color_continuous_scale=[[0, "#F87171"], [0.5, PANEL], [1, ACCENT_2]],
                        range_color=[0, 1],
                    )
                    fig_lang.update_layout(xaxis_title="Average similarity after round-trip", yaxis_title="")
                    st.plotly_chart(style_fig(fig_lang, height=90 + 35 * len(by_lang)), use_container_width=True)

                    st.markdown("---")
                    st.markdown("**Most fragile texts** — largest drift, worst-case language shown")
                    worst_per_text = df.loc[df.groupby("Text")["Similarity"].idxmin()].sort_values("Similarity")
                    for _, row in worst_per_text.head(3).iterrows():
                        st.write(f"- {row['Text']}  — {row['Language']}: {row['Similarity']:.3f}")
                        st.caption(f'Round-tripped: "{row["Round-tripped"]}"')

                    st.markdown("---")
                    with st.expander("Full results table"):
                        st.dataframe(full_df, use_container_width=True, hide_index=True)

                    st.markdown("---")
                    st.markdown("**Multi-model comparison — Semantic Robustness Leaderboard**")
                    st.markdown(
                        "<div class='section-caption'>Runs the same round-trip translations through "
                        "several embedding models and ranks them by how well each preserves meaning — "
                        "the translations above are reused, so only re-embedding is repeated per model.</div>",
                        unsafe_allow_html=True,
                    )
                    available_models = [
                        "all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-multilingual-MiniLM-L12-v2"
                    ]
                    model_choices = st.multiselect(
                        "Models to compare", available_models,
                        default=[model_name_used], key="rb_model_choices",
                    )
                    if st.button("Compare models") and model_choices:
                        target_codes = st.session_state.robustness_targets
                        source_code = st.session_state.robustness_source
                        total_mm = len(texts) * len(target_codes) * len(model_choices)
                        progress_mm = st.progress(0.0, text="Comparing models...")
                        mm_rows, done_mm = [], 0
                        for mdl in model_choices:
                            orig_vecs_mdl = embed_texts(tuple(texts), "Sentence Transformer", mdl)
                            for ti, text in enumerate(texts):
                                for lang_name, code in target_codes.items():
                                    try:
                                        _, back = round_trip_translate(text, code, source_code)
                                        back_vec = embed_single_neural(back, mdl)
                                        sim = float(cosine_similarity([orig_vecs_mdl[ti]], [back_vec])[0, 0])
                                        mm_rows.append({"Model": mdl, "Text": text, "Language": lang_name,
                                                         "Similarity": sim})
                                    except Exception:
                                        mm_rows.append({"Model": mdl, "Text": text, "Language": lang_name,
                                                         "Similarity": None})
                                    done_mm += 1
                                    progress_mm.progress(
                                        done_mm / total_mm, text=f"Comparing models... ({done_mm}/{total_mm})"
                                    )
                        progress_mm.empty()
                        st.session_state.multi_model_df = pd.DataFrame(mm_rows)

                    if st.session_state.get("multi_model_df") is not None:
                        mm_full = st.session_state.multi_model_df
                        mm_df = mm_full.dropna(subset=["Similarity"])
                        if mm_df.empty:
                            st.warning("No successful results — check your internet connection and try again.")
                        else:
                            leaderboard = mm_df.groupby("Model")["Similarity"].mean().sort_values(ascending=False)
                            fig_lb = px.bar(
                                x=leaderboard.values * 100, y=leaderboard.index, orientation="h",
                                color=leaderboard.values,
                                color_continuous_scale=[[0, "#F87171"], [0.5, PANEL], [1, ACCENT_2]],
                                range_color=[0, 1],
                            )
                            fig_lb.update_layout(
                                xaxis_title="Semantic Robustness Score (0-100)", yaxis_title="", showlegend=False
                            )
                            st.plotly_chart(style_fig(fig_lb, height=110 + 60 * len(leaderboard)),
                                             use_container_width=True)
                            st.caption(
                                f"Best: {leaderboard.index[0]} ({leaderboard.iloc[0] * 100:.1f}/100) · "
                                f"Worst: {leaderboard.index[-1]} ({leaderboard.iloc[-1] * 100:.1f}/100)"
                            )

    # --- Functorial Check ---
    with tab_functorial:
        st.markdown("**Functorial commutativity check**")
        st.markdown(
            "<div class='section-caption'>Tests whether two ways of combining translation and "
            "summarization arrive at the same meaning: translate-then-summarize vs. "
            "summarize-then-translate. If the two paths land close together in embedding space, the "
            "operations commute for this text and language — the order didn't matter. If they diverge, "
            "order changes meaning.</div>",
            unsafe_allow_html=True,
        )

        if not HAS_TRANSLATOR:
            st.caption("Install the `deep-translator` package to enable this feature.")
        elif not HAS_ANTHROPIC or not ai_key:
            st.caption(
                "Requires the `anthropic` package and an API key in the sidebar — summarization is "
                "done by Claude, translation is free."
            )
        elif embed_method_used != "Sentence Transformer":
            st.caption(
                "Requires the Sentence Transformer embedding method — select it in the sidebar and "
                "click Analyze again."
            )
        else:
            source_choice = st.selectbox(
                "Source language of the text", list(LANGUAGE_OPTIONS.keys()), index=0, key="fc_source"
            )
            target_pool_fc = [name for name in LANGUAGE_OPTIONS if name != source_choice]
            target_choice = st.selectbox(
                "Target language", target_pool_fc,
                index=target_pool_fc.index("Hindi") if "Hindi" in target_pool_fc else 0, key="fc_target",
            )

            source_options = ["Choose from your texts", "Type a custom text"]
            input_mode_fc = st.radio("Text to test", source_options, horizontal=True, key="fc_input_mode")
            if input_mode_fc == "Choose from your texts":
                fc_idx = st.selectbox(
                    "Text", range(len(texts)), format_func=lambda i: texts[i][:70], key="fc_text_idx"
                )
                fc_text = texts[fc_idx]
            else:
                fc_text = st.text_area(
                    "Custom text (a paragraph works better than a single short sentence, since "
                    "there's more to summarize)", key="fc_custom_text", height=120,
                )

            if st.button("Run commutativity check") and fc_text:
                try:
                    with st.spinner("Translating and summarizing along both paths..."):
                        result = functorial_commutativity_check(
                            ai_key, ai_model, fc_text,
                            LANGUAGE_OPTIONS[source_choice], source_choice,
                            LANGUAGE_OPTIONS[target_choice], target_choice,
                            model_name_used,
                        )
                    st.session_state.functorial_result = result
                except Exception as e:
                    st.error(f"Could not complete the check: {e}")

            if st.session_state.get("functorial_result"):
                r = st.session_state.functorial_result
                st.metric("Commutativity score", f"{r['score']:.3f}")
                st.caption(
                    "1.0 = the two paths arrive at identical meaning (fully commutes). "
                    "Lower values mean the order of translation and summarization changes the result."
                )

                st.markdown("---")
                colP1, colP2 = st.columns(2)
                with colP1:
                    st.markdown(f"**Path A — translate then summarize** (→ {target_choice} → back)")
                    st.write(f"1. Translated: {r['translated']}")
                    st.write(f"2. Summarized in {target_choice}: {r['summary_in_target']}")
                    st.write(f"3. Translated back: {r['path_a_result']}")
                with colP2:
                    st.markdown(f"**Path B — summarize then translate** (→ {target_choice} → back)")
                    st.write(f"1. Summarized in {source_choice}: {r['summary_in_source']}")
                    st.write(f"2. Translated to {target_choice}: {r['path_b_intermediate']}")
                    st.write(f"3. Translated back: {r['path_b_result']}")

    # --- Clusters ---
    with tab_clusters:
        for c in sorted(set(labels)):
            members = [t for t, l in zip(texts, labels) if l == c]
            terms = cluster_terms.get(c, [])
            representative = texts[medoids[c]]
            with st.expander(f"Cluster {c} — {len(members)} texts", expanded=(c == 0)):
                st.markdown(f"**Representative text:** {representative}")
                st.markdown(f"**Keywords:** {', '.join(terms) if terms else 'N/A'}")
                st.write("")
                for m in members:
                    st.write(f"- {m}")

        margins = cluster_confidence(embeddings, labels)
        borderline_idx = np.argsort(margins)[: min(5, len(texts))]
        st.markdown("---")
        with st.expander("Borderline texts — closest to switching clusters"):
            st.markdown(
                "<div class='section-caption'>These texts have the smallest gap between their assigned "
                "cluster and the next-closest one — the ones KMeans was least confident about.</div>",
                unsafe_allow_html=True,
            )
            for i in borderline_idx:
                st.write(f"- {texts[i]}  (cluster {labels[i]}, margin {margins[i]:.3f})")

        st.markdown("---")
        st.markdown("**AI-generated insights**")
        st.markdown(
            "<div class='section-caption'>Sends short cluster excerpts to Claude to generate a "
            "human-readable label and summary for each cluster, plus one line describing the dataset "
            "as a whole.</div>", unsafe_allow_html=True,
        )
        if not HAS_ANTHROPIC:
            st.caption("Install the `anthropic` package to enable this feature.")
        elif not ai_key:
            st.caption("Add an Anthropic API key in the sidebar to enable this feature.")
        else:
            if st.button("Generate AI insights"):
                clusters_dict = {c: [t for t, l in zip(texts, labels) if l == c] for c in sorted(set(labels))}
                try:
                    with st.spinner("Contacting Claude..."):
                        st.session_state.ai_insights = generate_ai_insights(ai_key, ai_model, clusters_dict)
                except Exception as e:
                    st.error(f"Could not generate insights: {e}")

            if st.session_state.get("ai_insights"):
                insights = st.session_state.ai_insights
                st.info(insights.get("overall_theme", ""))
                for item in insights.get("clusters", []):
                    st.markdown(f"**Cluster {item.get('id')} — {item.get('label')}:** {item.get('summary')}")

        st.markdown("---")
        st.markdown("**Cluster contrast finder**")
        st.markdown(
            "<div class='section-caption'>Pick two clusters and ask Claude to explain the single biggest "
            "conceptual difference between them, grounded in the actual texts.</div>", unsafe_allow_html=True,
        )
        if not HAS_ANTHROPIC or not ai_key:
            st.caption("Requires the `anthropic` package and an API key in the sidebar.")
        else:
            unique_clusters = sorted(set(labels))
            colX, colY = st.columns(2)
            cluster_a = colX.selectbox("Cluster A", unique_clusters, key="contrast_a")
            cluster_b = colY.selectbox(
                "Cluster B", unique_clusters, index=min(1, len(unique_clusters) - 1), key="contrast_b"
            )
            if st.button("Explain the difference") and cluster_a != cluster_b:
                texts_a = [t for t, l in zip(texts, labels) if l == cluster_a]
                texts_b = [t for t, l in zip(texts, labels) if l == cluster_b]
                try:
                    with st.spinner("Contacting Claude..."):
                        contrast = generate_cluster_contrast(
                            ai_key, ai_model, cluster_a, texts_a, cluster_b, texts_b
                        )
                    st.info(contrast)
                except Exception as e:
                    st.error(f"Could not generate contrast: {e}")

        st.markdown("---")
        st.markdown("**Diverse subset selection**")
        st.markdown(
            "<div class='section-caption'>Selects a small, non-redundant sample that covers the spread "
            "of the whole set — useful for curating training data or spot-checking a large corpus.</div>",
            unsafe_allow_html=True,
        )
        k_diverse = st.slider("Subset size", 2, min(10, len(texts)), min(4, len(texts)))
        if st.button("Select diverse subset"):
            idxs = mmr_diverse_subset(embeddings, k_diverse, sim_matrix)
            for rank, i in enumerate(idxs, 1):
                st.write(f"{rank}. {texts[i]}")

    # --- Insights ---
    with tab_insights:
        st.markdown("**Zero-shot semantic tagging**")
        st.markdown(
            "<div class='section-caption'>Provide candidate labels and every text is scored against each "
            "one by embedding similarity — no training data or fine-tuning required.</div>",
            unsafe_allow_html=True,
        )
        label_input = st.text_input(
            "Candidate labels (comma-separated)", placeholder="e.g. finance, nature, food, technology",
            key="zshot_labels",
        )
        if label_input:
            candidate_labels = [l.strip() for l in label_input.split(",") if l.strip()]
            if len(candidate_labels) >= 2:
                zs_scores = zero_shot_scores(texts, candidate_labels, embed_method_used, model_name_used)
                best_idx = zs_scores.argmax(axis=1)
                tag_df = pd.DataFrame({
                    "Text": [t[:70] + ("…" if len(t) > 70 else "") for t in texts],
                    "Best label": [candidate_labels[i] for i in best_idx],
                    "Confidence": zs_scores.max(axis=1),
                })
                st.dataframe(tag_df, use_container_width=True, hide_index=True)
            else:
                st.caption("Enter at least two candidate labels, separated by commas.")
        else:
            st.caption("Enter candidate labels above to tag every text.")

        st.markdown("---")
        st.markdown("**Corpus-wide term importance**")
        st.markdown(
            "<div class='section-caption'>The most central terms across the whole set, sized by average "
            "TF-IDF weight — a lightweight, non-redundant alternative to a raw word cloud.</div>",
            unsafe_allow_html=True,
        )
        term_vectorizer = TfidfVectorizer(stop_words="english", max_features=40)
        tfidf_all = term_vectorizer.fit_transform(texts)
        terms_all = term_vectorizer.get_feature_names_out()
        weights = np.asarray(tfidf_all.mean(axis=0)).ravel()
        if len(terms_all):
            rng = np.random.default_rng(42)
            term_df = pd.DataFrame({
                "term": terms_all, "weight": weights,
                "x": rng.uniform(0, 1, len(terms_all)), "y": rng.uniform(0, 1, len(terms_all)),
            })
            fig_terms = px.scatter(
                term_df, x="x", y="y", size="weight", text="term", color="weight",
                color_continuous_scale=[[0, PANEL], [1, ACCENT_2]], size_max=55,
            )
            fig_terms.update_traces(textposition="middle center", textfont=dict(color=TEXT, size=11))
            st.plotly_chart(style_fig(fig_terms, height=460, show_axes=False), use_container_width=True)
        else:
            st.caption("Not enough distinct terms to visualize.")

        st.markdown("---")
        st.markdown("**Coverage gap analysis**")
        st.markdown(
            "<div class='section-caption'>Finds regions of the embedding space with no nearby text — "
            "topics or angles your dataset doesn't currently cover, rather than just showing what it "
            "already has.</div>", unsafe_allow_html=True,
        )
        n_gaps = st.slider("Number of gaps to find", 2, 8, 4, key="n_gaps")
        if st.button("Find gaps"):
            gap_points, gap_dists = find_gap_regions(coords[:, :2], n_gaps=n_gaps)
            st.session_state.gap_points = gap_points
            st.session_state.gap_dists = gap_dists

        if st.session_state.get("gap_points") is not None:
            gap_points = st.session_state.gap_points
            gap_dists = st.session_state.gap_dists
            coords_2d = coords[:, :2]

            fig_gap = go.Figure()
            for c in sorted(set(labels)):
                idx = np.where(labels == c)[0]
                fig_gap.add_trace(go.Scatter(
                    x=coords_2d[idx, 0], y=coords_2d[idx, 1], mode="markers",
                    marker=dict(size=10, color=PALETTE[c % len(PALETTE)], line=dict(width=1, color=BG)),
                    text=[texts[i][:80] for i in idx], hoverinfo="text", name=f"Cluster {c}",
                ))
            fig_gap.add_trace(go.Scatter(
                x=gap_points[:, 0], y=gap_points[:, 1], mode="markers",
                marker=dict(size=17, color="#FBBF24", symbol="diamond", line=dict(width=2, color=BG)),
                text=[f"Gap {i + 1} — nearest text distance {gap_dists[i]:.2f}" for i in range(len(gap_points))],
                hoverinfo="text", name="Coverage gap",
            ))
            st.plotly_chart(style_fig(fig_gap, height=560), use_container_width=True)

            st.markdown("**Nearby context for each gap**")
            gap_contexts = []
            for gi, gp in enumerate(gap_points):
                dists_to_texts = np.linalg.norm(coords_2d - gp, axis=1)
                nearest_idx = np.argsort(dists_to_texts)[:3]
                nearby = [texts[i] for i in nearest_idx]
                gap_contexts.append({"id": gi, "nearby_texts": nearby})
                with st.expander(f"Gap {gi + 1}"):
                    for t in nearby:
                        st.write(f"- {t}")

            if not HAS_ANTHROPIC or not ai_key:
                st.caption("Add an Anthropic API key in the sidebar to get AI-generated suggestions for each gap.")
            elif st.button("Suggest what belongs in each gap"):
                try:
                    with st.spinner("Contacting Claude..."):
                        result = generate_gap_suggestions(ai_key, ai_model, gap_contexts)
                    for item in result.get("gaps", []):
                        st.info(f"Gap {item.get('id', 0) + 1}: {item.get('suggestion')}")
                except Exception as e:
                    st.error(f"Could not generate suggestions: {e}")
        else:
            st.caption('Click "Find gaps" to analyze coverage.')

    # --- Export ---
    with tab_export:
        out_df = pd.DataFrame({"text": texts, "cluster": labels})
        emb_df = pd.DataFrame(embeddings)
        emb_df.insert(0, "text", texts)

        csv_buf = io.StringIO()
        out_df.to_csv(csv_buf, index=False)
        emb_buf = io.StringIO()
        emb_df.to_csv(emb_buf, index=False)

        colx, coly = st.columns(2)
        colx.download_button("Download clusters (CSV)", csv_buf.getvalue(), "clusters.csv", "text/csv",
                              use_container_width=True)
        coly.download_button("Download embeddings (CSV)", emb_buf.getvalue(), "embeddings.csv", "text/csv",
                              use_container_width=True)

        json_payload = json.dumps(
            {"texts": texts, "clusters": labels.tolist(), "similarity_matrix": sim_matrix.tolist()},
            indent=2,
        )
        st.download_button("Download full results (JSON)", json_payload, "semantic_atlas_results.json",
                            "application/json", use_container_width=True)

        st.markdown("---")
        st.markdown("**Offline report**")
        st.markdown(
            "<div class='section-caption'>A single self-contained HTML file with the heatmap, embedding "
            "plot, and full cluster breakdown baked in — viewable in any browser, no Streamlit required.</div>",
            unsafe_allow_html=True,
        )
        report_html = build_offline_report_html(
            texts, labels, sim_matrix, cluster_terms, medoids, k_final, sil, embed_method_used, coords
        )
        st.download_button(
            "Download offline report (HTML)", report_html, "semantic_atlas_report.html",
            "text/html", use_container_width=True,
        )

else:
    render_welcome()

