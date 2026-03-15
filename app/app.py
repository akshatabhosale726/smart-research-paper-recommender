import streamlit as st
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from model.recommender import recommend_papers

st.set_page_config(
    page_title="Smart Research Paper Recommender",
    layout="wide"
)

st.title("📚 Smart Research Paper Recommendation System")

st.write("🔍 Discover the most relevant and latest research papers.")

query = st.text_input("🔍 Enter research topic")

if st.button("Search Papers"):

    if query.strip() == "":
        st.warning("Please enter a research topic")

    else:

        papers = recommend_papers(query)

        st.subheader("Top Relevant Papers")

        for paper in papers:

            st.markdown(f"## {paper['title']}")

            col1, col2 = st.columns(2)

            col1.metric("📊 Relevance Score", f"{paper['score']} %")
            col2.metric("📅 Year", paper["year"])

            if paper["authors"]:
                st.markdown(f"**👨‍🔬 Authors:** {paper['authors']}")

            if paper["category"]:
                st.markdown(f"**🏷 Category:** {paper['category']}")

            st.markdown("### 📄 Abstract / Summary")
            st.write(paper["summary"])

            st.markdown("### ⚠ Drawbacks")
            st.write(paper["drawbacks"])

            st.markdown("### 🚀 Future Scope")
            st.write(paper["future_scope"])

            st.markdown("### 🔗 Links")

            st.markdown(
                f'<a href="{paper["paper_link"]}" target="_blank">🔗 Original Paper Page</a>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<a href="{paper["pdf_link"]}" target="_blank">📄 View PDF</a>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<a href="{paper["scholar"]}" target="_blank">📈 Google Scholar</a>',
                unsafe_allow_html=True
            )

            st.markdown("---")
