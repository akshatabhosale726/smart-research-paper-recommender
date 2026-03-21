import requests
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.analyzer import extract_drawbacks, future_scope


def fetch_arxiv(query, max_results=20):
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"

    response = requests.get(url, timeout=10)
    root = ET.fromstring(response.content)

    papers = []

    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):

        title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()

        summary = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip().replace("\n", " ")

        authors = [
            author.find("{http://www.w3.org/2005/Atom}name").text
            for author in entry.findall("{http://www.w3.org/2005/Atom}author")
        ]

        published = entry.find("{http://www.w3.org/2005/Atom}published").text
        year = int(published[:4])

        link = entry.find("{http://www.w3.org/2005/Atom}id").text
        pdf_link = link.replace("abs", "pdf")

        papers.append({
            "title": title,
            "summary": summary,
            "authors": ", ".join(authors),
            "year": year,
            "link": link,
            "pdf": pdf_link
        })

    return papers


def recommend_papers(query, top_n=5):

    if not query.strip():
        return []

    try:
        papers = fetch_arxiv(query, max_results=25)

        if not papers:
            return []

        # -----------------------------
        # TF-IDF on REAL abstracts
        # -----------------------------
        texts = [p["summary"] for p in papers]

        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(texts)

        query_vec = vectorizer.transform([query])
        similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()

        # -----------------------------
        # SCORING (SMART)
        # -----------------------------
        results = []

        for i, paper in enumerate(papers):

            sim_score = similarity[i]

            # normalize year (recent = better)
            year_score = (paper["year"] - 2015) / 10

            final_score = (0.7 * sim_score) + (0.3 * year_score)

            results.append((i, final_score))

        # sort
        results = sorted(results, key=lambda x: x[1], reverse=True)[:top_n]

        max_score = results[0][1] if results else 1

        final_papers = []

        for i, score_val in results:

            p = papers[i]

            score = round((score_val / max_score) * 100, 2)

            final_papers.append({
                "title": p["title"],
                "score": score,
                "authors": p["authors"],
                "year": p["year"],
                "category": "arXiv",

                "summary": p["summary"][:600] + "...",

                "drawbacks": extract_drawbacks(p["summary"]),
                "future_scope": future_scope(),

                "paper_link": p["link"],
                "pdf_link": p["pdf"],
                "scholar": f"https://scholar.google.com/scholar?q={p['title'].replace(' ', '+')}"
            })

        return final_papers

    except Exception as e:
        print("ERROR:", e)
        return []
