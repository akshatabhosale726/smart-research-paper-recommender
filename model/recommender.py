import pandas as pd
import requests
import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.analyzer import extract_drawbacks, future_scope


# -----------------------------
# LOAD DATA (SAFE)
# -----------------------------
def load_data():
    file_id = "1Rz06PQsTbRN9FnijXxrj2SThnL1NZiux"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    output = "dataset.csv"

    try:
        if not os.path.exists(output):
            print("⬇ Downloading dataset...")

            session = requests.Session()
            response = session.get(url, stream=True)

            for key, value in response.cookies.items():
                if key.startswith("download_warning"):
                    url = f"https://drive.google.com/uc?export=download&confirm={value}&id={file_id}"
                    response = session.get(url, stream=True)
                    break

            with open(output, "wb") as f:
                for chunk in response.iter_content(1024):
                    if chunk:
                        f.write(chunk)

        df = pd.read_csv(output, low_memory=False)
        print("✅ Dataset Loaded:", df.shape)
        return df

    except Exception as e:
        print("❌ Dataset failed:", e)

        # fallback (NEVER FAIL)
        return pd.DataFrame({
            "title": ["AI Automation"],
            "abstract": ["Artificial intelligence is used for automation and machine learning."]
        })


# -----------------------------
# INIT DATA
# -----------------------------
df = load_data()
df.columns = df.columns.str.lower().str.strip()

# -----------------------------
# COLUMN DETECTION
# -----------------------------
title_col = next((c for c in df.columns if "title" in c), df.columns[0])

text_col = next((c for c in df.columns 
                 if any(x in c for x in ["abstract", "summary", "text", "description"])), None)

if not text_col:
    text_col = df.columns[0]

author_col = next((c for c in df.columns if "author" in c), None)
date_col = next((c for c in df.columns if "date" in c or "year" in c), None)
category_col = next((c for c in df.columns if "category" in c), None)

print("Using:", title_col, text_col)

# -----------------------------
# CLEAN TEXT (SAFE)
# -----------------------------
def clean_text(text):
    text = str(text)
    if len(text.strip()) < 5:
        return ""
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

df[text_col] = df[text_col].fillna("").astype(str).apply(clean_text)
df[title_col] = df[title_col].fillna("").astype(str)

# -----------------------------
# FIX EMPTY DATA
# -----------------------------
df = df[df[text_col] != ""]

if len(df) < 10:
    print("⚠️ Too little data → using fallback dataset")

    df = pd.DataFrame({
        "title": [
            "AI Automation Systems",
            "Deep Learning for Automation",
            "Machine Learning in Industry"
        ],
        "abstract": [
            "AI is used in automation systems for efficiency.",
            "Deep learning improves automation accuracy.",
            "Machine learning enables smart industry automation."
        ]
    })

    title_col = "title"
    text_col = "abstract"

# -----------------------------
# DATE FIX
# -----------------------------
if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

# -----------------------------
# LIMIT SIZE (FAST)
# -----------------------------
df = df.sample(n=min(4000, len(df)), random_state=42)

print("Final dataset:", df.shape)

# -----------------------------
# TF-IDF SAFE BUILD
# -----------------------------
vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)

try:
    tfidf_matrix = vectorizer.fit_transform(df[text_col])
except Exception as e:
    print("⚠️ TF-IDF failed → using simple fallback text")

    df[text_col] = df[text_col].apply(lambda x: x if len(x) > 10 else "machine learning ai data")

    tfidf_matrix = vectorizer.fit_transform(df[text_col])

# -----------------------------
# RECOMMEND FUNCTION
# -----------------------------
def recommend_papers(query, top_n=5):

    if not query.strip():
        return []

    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_indices = similarity.argsort()[-top_n:][::-1]

    papers = []

    for i in top_indices:
        row = df.iloc[i]

        title = str(row[title_col])
        summary = str(row[text_col])[:400] + "..."

        authors = str(row[author_col]) if author_col else "Unknown"
        category = str(row[category_col]) if category_col else "Research"

        year = "Unknown"
        if date_col and pd.notna(row.get(date_col)):
            try:
                year = pd.to_datetime(row[date_col]).year
            except:
                pass

        score = round(similarity[i] * 100, 2)

        search_title = title.replace(" ", "+")

        papers.append({
            "title": title,
            "score": score,
            "authors": authors,
            "year": year,
            "category": category,
            "summary": summary,

            "drawbacks": extract_drawbacks(summary),
            "future_scope": future_scope(),

            "paper_link": f"https://arxiv.org/search/?query={search_title}&searchtype=all",
            "pdf_link": f"https://arxiv.org/search/?query={search_title}&searchtype=all",
            "scholar": f"https://scholar.google.com/scholar?q={search_title}"
        })

    return papers
