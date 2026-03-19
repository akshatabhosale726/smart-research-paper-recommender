import pandas as pd
import os

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.analyzer import extract_drawbacks, future_scope

DRIVE_URL = "https://drive.google.com/uc?id=1Rz06PQsTbRN9FnijXxrj2SThnL1NZiux"

try:
    df = pd.read_csv(DRIVE_URL)
except Exception as e:
    raise Exception("❌ Failed to load dataset. Make sure Google Drive file is PUBLIC.")

# -----------------------------
# DEBUG: PRINT COLUMNS
# -----------------------------
print("Dataset Columns:", df.columns)

df.columns = df.columns.str.lower().str.strip()

possible_title_cols = ["title", "titles", "paper_title", "name"]

title_col = None
for col in possible_title_cols:
    if col in df.columns:
        title_col = col
        break

if title_col is None:
    title_col = df.columns[0]  
    
possible_text_cols = ["abstract", "summary", "summaries", "description", "content"]

text_col = None
for col in possible_text_cols:
    if col in df.columns:
        text_col = col
        break

if text_col is None:
    text_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

# -----------------------------
# OPTIONAL COLUMNS
# -----------------------------
author_col = "authors" if "authors" in df.columns else None
date_col = "published_date" if "published_date" in df.columns else None
category_col = "categories" if "categories" in df.columns else None

df[text_col] = df[text_col].fillna("").astype(str)
df[title_col] = df[title_col].fillna("").astype(str)

if author_col:
    df[author_col] = df[author_col].fillna("Unknown")

if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df[text_col])

def extract_keywords(text):
    try:
        cv = CountVectorizer(stop_words="english", max_features=5)
        return ", ".join(cv.fit([text]).get_feature_names_out())
    except:
        return "N/A"

def recommend_papers(query, top_n=5):

    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_indices = similarity.argsort()[-100:]

    results = []

    for i in top_indices:
        row = df.iloc[i]
        sim_score = similarity[i]

        year = row[date_col].year if date_col and pd.notna(row[date_col]) else 2000
        recency_score = (year - 2000) / 30

        final_score = (0.65 * sim_score) + (0.35 * recency_score)

        results.append((i, final_score))

    results = sorted(results, key=lambda x: x[1], reverse=True)[:top_n]

    if not results:
        return []

    max_score = results[0][1] if results[0][1] != 0 else 1

    papers = []

    for i, score_val in results:

        row = df.iloc[i]

        title = row[title_col]
        summary = row[text_col][:500] + "..."

        authors = row[author_col] if author_col else "Unknown"
        category = row[category_col] if category_col else "Research"
        year = row[date_col].year if date_col and pd.notna(row[date_col]) else "Unknown"

        score = round((score_val / max_score) * 100, 2)

        search_title = str(title).replace(" ", "+")

        papers.append({
            "title": title,
            "score": score,
            "authors": authors,
            "year": year,
            "category": category,
            "summary": summary,
            "keywords": extract_keywords(summary),

            "drawbacks": extract_drawbacks(summary),
            "future_scope": future_scope(),

            "paper_link": f"https://arxiv.org/search/?query={search_title}&searchtype=all",
            "pdf_link": f"https://arxiv.org/search/?query={search_title}&searchtype=all",
            "scholar": f"https://scholar.google.com/scholar?q={search_title}"
        })

    return papers
