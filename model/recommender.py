import pandas as pd
import os
import requests
from io import StringIO

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.analyzer import extract_drawbacks, future_scope

def load_data():
    try:
        file_id = "1Rz06PQsTbRN9FnijXxrj2SThnL1NZiux"

        url = f"https://drive.google.com/uc?id={file_id}"

        response = requests.get(url)
        response.raise_for_status()

        df = pd.read_csv(StringIO(response.text))

        print("✅ Dataset loaded successfully")
        print("Columns:", df.columns)

        return df

    except Exception as e:
        raise Exception(f"❌ Error loading dataset: {e}")


df = load_data()
df.columns = df.columns.str.lower().str.strip()

print("Cleaned Columns:", df.columns)
title_col = None
text_col = None

print("Available Columns:", df.columns)

# Detect title column
title_col = None
for col in df.columns:
    if "title" in col.lower():
        title_col = col
        break

# Detect text column (more powerful logic)
text_col = None

priority_keywords = ["abstract", "summary", "text", "content", "description"]

for key in priority_keywords:
    for col in df.columns:
        if key in col.lower():
            text_col = col
            break
    if text_col:
        break
        
if not text_col:
    print("⚠️ No standard text column found, selecting largest text column...")

    text_lengths = {}
    for col in df.columns:
        try:
            avg_len = df[col].astype(str).str.len().mean()
            text_lengths[col] = avg_len
        except:
            continue

    text_col = max(text_lengths, key=text_lengths.get)

# FINAL CHECK
if not title_col:
    title_col = df.columns[0]  # fallback

print("✅ Selected title column:", title_col)
print("✅ Selected text column:", text_col)

author_col = next((c for c in df.columns if "author" in c), None)
date_col = next((c for c in df.columns if "date" in c or "year" in c), None)
category_col = next((c for c in df.columns if "category" in c), None)

df[text_col] = df[text_col].fillna("").astype(str)
df[title_col] = df[title_col].fillna("").astype(str)

# Remove empty rows
df = df[df[text_col].str.strip() != ""]

if df.empty:
    raise Exception("❌ Dataset has no valid text data after cleaning")

print("Final dataset size:", df.shape)

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
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

        # Year handling
        year = 2000
        if date_col and pd.notna(row[date_col]):
            try:
                year = pd.to_datetime(row[date_col]).year
            except:
                pass

        recency_score = (year - 2000) / 30
        final_score = (0.65 * sim_score) + (0.35 * recency_score)

        results.append((i, final_score))

    results = sorted(results, key=lambda x: x[1], reverse=True)[:top_n]

    max_score = results[0][1] if results else 1

    papers = []

    for i, score_val in results:
        row = df.iloc[i]

        title = row[title_col]
        summary = row[text_col][:500] + "..."

        authors = row[author_col] if author_col else "Unknown"
        category = row[category_col] if category_col else "Research"

        year = "Unknown"
        if date_col and pd.notna(row[date_col]):
            try:
                year = pd.to_datetime(row[date_col]).year
            except:
                pass

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
