import pandas as pd
import requests
from io import StringIO

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.analyzer import extract_drawbacks, future_scope
def load_data():
    file_id = "1Rz06PQsTbRN9FnijXxrj2SThnL1NZiux"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()

        df = pd.read_csv(StringIO(response.text), low_memory=False)

        print("✅ Dataset Loaded Successfully")
        print("Columns:", df.columns)
        print("Shape:", df.shape)

        return df

    except Exception as e:
        print("❌ ERROR loading dataset:", e)

        # fallback minimal dataset
        return pd.DataFrame({
            "title": ["AI Automation Example"],
            "abstract": ["This paper explains automation using artificial intelligence."]
        })


df = load_data()
df.columns = df.columns.str.lower().str.strip()

title_col = next((c for c in df.columns if "title" in c), None)
text_col = next((c for c in df.columns if "abstract" in c or "summary" in c), None)
author_col = next((c for c in df.columns if "author" in c), None)
date_col = next((c for c in df.columns if "date" in c or "year" in c), None)
category_col = next((c for c in df.columns if "category" in c), None)

if not title_col:
    title_col = df.columns[0]

if not text_col:
    text_col = max(df.columns, key=lambda c: df[c].astype(str).str.len().mean())

print("🧠 Using:")
print("Title:", title_col)
print("Text:", text_col)
import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)   
    text = re.sub(r'\s+', ' ', text).strip() 
    return text

df[text_col] = df[text_col].fillna("").astype(str).apply(clean_text)
df[title_col] = df[title_col].fillna("").astype(str)

df = df[df[text_col].str.len() > 20]

if df.empty:
    print("⚠️ Dataset empty after cleaning → using fallback")

    df = pd.DataFrame({
        "title": ["AI Automation"],
        "abstract": ["Artificial intelligence is used for automation and machine learning tasks."]
    })

    title_col = "title"
    text_col = "abstract"

print("✅ Cleaned dataset size:", df.shape)
df = df.sample(n=min(8000, len(df)), random_state=42)

print("📊 Final dataset size:", df.shape)

vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
tfidf_matrix = vectorizer.fit_transform(df[text_col])

def recommend_papers(query, top_n=5):

    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # get best indices
    top_indices = similarity.argsort()[-top_n*5:]

    results = []

    for i in top_indices:
        row = df.iloc[i]
        sim_score = similarity[i]

        # year handling
        year = 2015
        if date_col and pd.notna(row.get(date_col)):
            try:
                year = pd.to_datetime(row[date_col]).year
            except:
                pass

        recency_score = (year - 2000) / 30
        final_score = (0.7 * sim_score) + (0.3 * recency_score)

        results.append((i, final_score))

    # sort
    results = sorted(results, key=lambda x: x[1], reverse=True)[:top_n]

    if not results:
        return []

    max_score = results[0][1] if results[0][1] != 0 else 1

    papers = []

    for i, score_val in results:
        row = df.iloc[i]

        title = str(row[title_col])
        summary = str(row[text_col])[:400] + "..."

        authors = row[author_col] if author_col else "Unknown"
        category = row[category_col] if category_col else "Research"

        year = "Unknown"
        if date_col and pd.notna(row.get(date_col)):
            try:
                year = pd.to_datetime(row[date_col]).year
            except:
                pass

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
