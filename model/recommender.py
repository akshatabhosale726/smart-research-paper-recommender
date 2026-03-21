import pandas as pd
import requests
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.analyzer import extract_drawbacks, future_scope

# -----------------------------
# FETCH DATA FROM arXiv API
# -----------------------------
def fetch_arxiv_data(query="machine learning", max_results=200):

    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"

    response = requests.get(url, timeout=15)
    root = ET.fromstring(response.content)

    data = []

    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):

        title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
        summary = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
        published = entry.find("{http://www.w3.org/2005/Atom}published").text.strip()

        authors = [
            author.find("{http://www.w3.org/2005/Atom}name").text
            for author in entry.findall("{http://www.w3.org/2005/Atom}author")
        ]

        link = entry.find("{http://www.w3.org/2005/Atom}id").text

        data.append({
            "title": title,
            "abstract": summary,
            "authors": ", ".join(authors),
            "year": int(published[:4]),
            "link": link
        })

    return pd.DataFrame(data)


# -----------------------------
# LOAD DATA (DYNAMIC PER QUERY)
# -----------------------------
def load_data(query):
    try:
        df = fetch_arxiv_data(query=query, max_results=200)
        print("✅ Data fetched:", df.shape)
        return df

    except Exception as e:
        print("❌ API failed, using fallback")

        return pd.DataFrame({
            "title": ["AI Automation Example"],
            "abstract": ["Artificial intelligence is used in automation systems."],
            "authors": ["Unknown"],
            "year": [2024],
            "link": ["https://arxiv.org"]
        })


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def recommend_papers(query, top_n=5):

    df = load_data(query)

    # Clean
    df["abstract"] = df["abstract"].fillna("")
    df = df[df["abstract"].str.len() > 50]

    if df.empty:
        return []

    # -----------------------------
    # SIMPLE ML (TF-IDF)
    # -----------------------------
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["abstract"])

    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # -----------------------------
    # SCORING (SIMPLE + EFFECTIVE)
    # -----------------------------
    results = []

    for i in range(len(df)):
        sim_score = similarity[i]

        year = df.iloc[i]["year"]
        recency_score = (year - 2015) / 10   # simple scaling

        final_score = (0.8 * sim_score) + (0.2 * recency_score)

        results.append((i, final_score))

    # sort best papers
    results = sorted(results, key=lambda x: x[1], reverse=True)[:top_n]

    max_score = results[0][1] if results[0][1] != 0 else 1

    papers = []

    for i, score_val in results:
        row = df.iloc[i]

        score = round((score_val / max_score) * 100, 2)

        papers.append({
            "title": row["title"],
            "score": score,
            "authors": row["authors"],
            "year": row["year"],
            "category": "arXiv",
            "summary": row["abstract"][:500] + "...",

            "drawbacks": extract_drawbacks(row["abstract"]),
            "future_scope": future_scope(),

            "paper_link": row["link"],
            "pdf_link": row["link"].replace("abs", "pdf"),
            "scholar": f"https://scholar.google.com/scholar?q={row['title'].replace(' ', '+')}"
        })

    return papers
