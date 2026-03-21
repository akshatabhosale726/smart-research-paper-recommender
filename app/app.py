import streamlit as st
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from model.recommender import recommend_papers

st.set_page_config(page_title="Smart Research Paper Recommender")

st.title("📚 Smart Research Paper Recommendation System")
st.write("🔍 Discover latest and most relevant research papers")

query = st.text_input("🔍 Enter research topic")

if st.button("Search"):

    results = recommend_papers(query)

    if not results:
        st.error("No results found. Try different query.")
    else:
        st.subheader("Top Relevant Papers")

        for paper in results:
            st.markdown(f"### {paper['title']}")

            col1, col2 = st.columns(2)
            col1.metric("📊 Score", f"{paper['score']}%")
            col2.metric("📅 Year", paper['year'])

            st.write(f"👨‍🔬 Authors: {paper['authors']}")
            st.write(f"🏷 Category: {paper['category']}")

            st.write("📄 Abstract")
            st.write(paper["summary"])

            st.write("⚠ Drawbacks")
            st.write(paper["drawbacks"])

            st.write("🚀 Future Scope")
            st.write(paper["future_scope"])

            st.markdown(f"[🔗 Paper]({paper['paper_link']}) | [📄 PDF]({paper['pdf_link']}) | [📈 Scholar]({paper['scholar']})")

            st.write("---")
