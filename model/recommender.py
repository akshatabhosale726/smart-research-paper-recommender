import pandas as pd
import requests
import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.analyzer import extract_drawbacks, future_scope

def load_data():
    file_id = "1Rz06PQsTbRN9FnijXxrj2SThnL1NZiux"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    output = "dataset.csv"

    try:
        if not os.path.exists(output):
            print("⬇ Downloading dataset...")

            session = requests.Session()
            response = session.get(url, stream=True)

            # Handle large file confirmation
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

        print("✅ DATASET LOADED")
        print("Columns:", df.columns)
        print("Shape:", df.shape)

        return df

    except Exception as e:
        raise Exception(f"❌ Dataset loading failed: {e}")

df.columns = df.columns.str.lower().str.strip()

title_col = next((c for c in df.columns if "title" in c), None)

text_col = next((c for c in df.columns 
                 if any(x in c for x in ["abstract", "summary", "description", "text"])), None)

author_col = next((c for c in df.columns 
                   if any(x in c for x in ["author", "creator"])), None)

date_col = next((c for c in df.columns 
                 if any(x in c for x in ["date", "year", "publish"])), None)

category_col = next((c for c in df.columns if "category" in c), None)

# fallback detection
if not title_col:
    title_col = df.columns[0]

if not text_col:
    text_col = max(df.columns, key=lambda c: df[c].astype(str).str.len().mean())

print("🧠 Using Columns:")
print("Title:", title_col)
print("Text:", text_col)
print("Author:", author_col)
print("Date:", date_col)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df[text_col] = df[text_col].fillna("").astype(str).apply(clean_text)
df[title_col] = df[title_col].fillna("").astype(str)

df = df[df[text_col].str.len() > 30]

if df.empty:
    raise Exception("❌ Dataset empty after cleaning")

if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

df = df.sample(n=min(6000, len(df)), random_state=42)

print("📊 Final dataset size:", df.shape)

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=3000,
    min_df=2
)

tfidf_matrix = vectorizer.fit_transform(df[text_col])

def recommend_papers(query, top_n=5):

    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_indices = similarity.argsort()[-top_n * 5:]

    results = []

    for i in top_indices:
        row = df.iloc[i]
        sim_score = similarity[i]

        year = 2018
        if date_col and pd.notna(row[date_col]):
            year = row[date_col].year

        recency_score = (year - 2000) / 30
        final_score = (0.7 * sim_score) + (0.3 * recency_score)

        results.append((i, final_score))

    results = sorted(results, key=lambda x: x[1], reverse=True)[:top_n]

    if not results:
        return []

    max_score = results[0][1] if results[0][1] != 0 else 1

    papers = []

    for i, score_val in results:
        row = df.iloc[i]

        title = str(row[title_col])

        summary = str(row[text_col])
        if len(summary) > 500:
            summary = summary[:500] + "..."

        authors = str(row[author_col]) if author_col else "Unknown"
        category = str(row[category_col]) if category_col else "Research"

        year = "Unknown"
        if date_col and pd.notna(row[date_col]):
            year = row[date_col].year

        score = round((score_val / max_score) * 100, 2)

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
