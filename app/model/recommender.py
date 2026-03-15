import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.analyzer import extract_drawbacks, future_scope, suggest_solution

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "arxiv_scientific dataset.csv")

df = pd.read_csv(DATA_PATH)

# make column names lowercase
df.columns = df.columns.str.lower()

if "title" in df.columns:
    title_col = "title"
elif "titles" in df.columns:
    title_col = "titles"
else:
    raise Exception("No title column found")

if "abstract" in df.columns:
    text_col = "abstract"
elif "summary" in df.columns:
    text_col = "summary"
elif "summaries" in df.columns:
    text_col = "summaries"
else:
    raise Exception("No abstract/summary column found")

author_col = "authors" if "authors" in df.columns else None
date_col = "published_date" if "published_date" in df.columns else None
category_col = "categories" if "categories" in df.columns else None

# -----------------------------
# CLEAN DATA
# -----------------------------
df[text_col] = df[text_col].fillna("")
df[title_col] = df[title_col].fillna("")

if author_col:
    df[author_col] = df[author_col].fillna("Unknown")

# convert date column
if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

# -----------------------------
# BUILD TF-IDF MODEL
# -----------------------------
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df[text_col])

def extract_keywords(text):

    try:
        cv = CountVectorizer(stop_words="english", max_features=5)
        words = cv.fit([text]).get_feature_names_out()
        return ", ".join(words)
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

        # normalize recency
        recency_score = (year - 2000) / 30

        # combine relevance + recency
        final_score = (0.65 * sim_score) + (0.35 * recency_score)

        results.append((i, final_score))

    # sort results
    results = sorted(results, key=lambda x: x[1], reverse=True)[:top_n]

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

        search_title = title.replace(" ", "+")

        keywords = extract_keywords(summary)

        papers.append({

            "title": title,
            "score": score,
            "authors": authors,
            "year": year,
            "category": category,
            "summary": summary,
            "keywords": keywords,

            "drawbacks": extract_drawbacks(summary),
            "future_scope": future_scope(),

            "paper_link": f"https://arxiv.org/search/?query={search_title}&searchtype=all",
            "pdf_link": f"https://arxiv.org/search/?query={search_title}&searchtype=all",
            "scholar": f"https://scholar.google.com/scholar?q={search_title}"

        })

    return papers
  
