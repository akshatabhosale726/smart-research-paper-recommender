"""
Smart Research Paper Recommendation System - Main Streamlit Application
Uses TF-IDF + Cosine Similarity + Recency Scoring (Simple ML only)
"""

import streamlit as st
import sys
import os
import time

# ── Setup project path ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from model.recommender import load_and_prepare_data, recommend_papers
from utils.analyzer import get_domain_list, get_trend_badge

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Research Paper Recommender",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# MINIMAL CSS - Only for small enhancements (no card HTML)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
div[data-testid="stExpander"] { border: 1px solid rgba(108,99,255,0.2); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING WITH CACHING
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def init_system():
    """Load dataset and build TF-IDF matrix (cached for performance)."""
    return load_and_prepare_data()


# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.title("🔬 Smart Research Paper Recommender")

# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────
try:
    with st.spinner("🔄 Loading research papers dataset & building TF-IDF model..."):
        df, tfidf_matrix, vectorizer, col_info = init_system()
    data_loaded = True
except Exception as e:
    st.error(f"❌ Failed to load dataset: {str(e)}")
    st.info("Please ensure a CSV dataset file exists in the 'dataset' folder.")
    data_loaded = False


if data_loaded:
    # ──────────────────────────────────────────────────────────────────────
    # SIDEBAR
    # ──────────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        domain = st.selectbox(
            "🏷️ Filter by Research Domain",
            get_domain_list(),
            index=0,
        )

        num_results = 10

        st.divider()
        st.header("📊 Dataset Info")
        c1, c2 = st.columns(2)
        c1.metric("📄 Papers", f"{len(df):,}")
        cats = df[col_info['category']].nunique() if col_info['category'] else 0
        c2.metric("📂 Categories", cats)

        st.divider()
        st.header("💡 Try These Topics")
        for topic in ["Machine Learning", "Cybersecurity", "NLP",
                       "Computer Vision", "Robotics", "Quantum Computing"]:
            st.code(topic, language=None)

    # ──────────────────────────────────────────────────────────────────────
    # SEARCH BAR
    # ──────────────────────────────────────────────────────────────────────
    query = st.text_input(
        "🔍 Enter your research topic",
        placeholder="e.g. Machine Learning, AI Automation, Blockchain, NLP...",
    )
    search_clicked = st.button("🚀 Search Papers", use_container_width=True, type="primary")

    # ──────────────────────────────────────────────────────────────────────
    # RESULTS
    # ──────────────────────────────────────────────────────────────────────
    if search_clicked or query:
        if not query or not query.strip():
            st.warning("⚠️ Please enter a research topic to search.")
        else:
            start_time = time.time()

            with st.spinner(f"🔍 Searching for papers on **{query}**..."):
                papers = recommend_papers(
                    query, df, tfidf_matrix, vectorizer, col_info,
                    domain=domain, top_n=num_results
                )

            elapsed = round(time.time() - start_time, 2)

            if not papers:
                st.warning("😔 No matching papers found. Try a different topic or broader search terms.")
            else:
                st.success(f"📄 Found {len(papers)} papers for \"{query}\" in {elapsed}s  |  Domain: {domain}")

                # ── Render each paper using native Streamlit components ──
                for paper in papers:

                    # ── Top Pick highlight ──
                    if paper['is_top']:
                        st.subheader(f"⭐ #{paper['rank']}  {paper['title']}")
                    else:
                        st.subheader(f"#{paper['rank']}  {paper['title']}")

                    # ── Badges row ──
                    badge_cols = st.columns([1, 1, 1, 1, 2])
                    badge_cols[0].caption(f"📂 {paper['category']}" if paper['category'] else "📂 Research")
                    badge_cols[1].caption(f"📅 {paper['year']}")
                    badge_cols[2].caption(f"{paper['trend_emoji']} {paper['trend_label']}")
                    badge_cols[3].caption(f"🎯 {paper['relevance_score']}% match")
                    if paper['is_top']:
                        badge_cols[4].caption("🏆 Top Pick")

                    # ── Authors ──
                    st.markdown(f"**👨‍🔬 Authors:** {paper['authors']}")

                    # ── Score Progress Bars ──
                    score_cols = st.columns(3)
                    with score_cols[0]:
                        st.caption("📊 Relevance Score")
                        st.progress(min(int(paper['relevance_score']), 100))
                        st.write(f"**{paper['relevance_score']}%**")
                    with score_cols[1]:
                        st.caption("🎯 Similarity Score")
                        st.progress(min(int(paper['similarity_score']), 100))
                        st.write(f"**{paper['similarity_score']}%**")
                    with score_cols[2]:
                        st.caption("📅 Recency Score")
                        st.progress(min(int(paper['recency_score']), 100))
                        st.write(f"**{paper['recency_score']}%**")

                    # ── Keywords ──
                    if paper['keywords']:
                        kw_text = " | ".join(paper['keywords'][:5])
                        st.markdown(f"**🔑 Keywords:** {kw_text}")

                    # ── Abstract (expandable) ──
                    with st.expander("📖 Read Abstract"):
                        st.write(paper['abstract'])

                    # ── Drawbacks & Future Scope ──
                    col_d, col_f = st.columns(2)
                    with col_d:
                        with st.expander("⚠️ Drawbacks"):
                            st.write(paper.get('drawbacks', 'Not available'))
                    with col_f:
                        with st.expander("🚀 Future Scope"):
                            st.write(paper.get('future_scope', 'Not available'))

                    # ── Links ──
                    link_cols = st.columns(3)
                    link_cols[0].link_button("📄 arXiv Paper", paper['arxiv_link'], use_container_width=True)
                    link_cols[1].link_button("📥 View PDF", paper['pdf_link'], use_container_width=True)
                    link_cols[2].link_button("🎓 Google Scholar", paper['scholar_link'], use_container_width=True)

                    st.divider()