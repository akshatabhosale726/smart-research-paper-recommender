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

        url = f"https://drive.google.com/uc?export=download&id={file_id}"

        response = requests.get(url)
        response.raise_for_status()

        print("🔍 Raw data preview:\n", response.text[:500])

        try:
            df = pd.read_csv(StringIO(response.text))
        except:
            df = pd.read_csv(StringIO(response.text), sep=';')

        print("✅ Dataset loaded")
        print("Columns:", df.columns)
        print("Shape:", df.shape)

        return df

    except Exception as e:
        raise Exception(f"❌ Error loading dataset: {e}")


df = load_data()

df.columns = df.columns.str.lower().str.strip()

print("🧹 Cleaned Columns:", df.columns)

title_col = None
text_col = None

# Detect title
for col in df.columns:
    if "title" in col:
        title_col = col
        break

# Detect text
priority_keywords = ["abstract", "summary", "text", "content", "description"]

for key in priority_keywords:
    for col in df.columns:
        if key in col:
            text_col = col
            break
    if text_col:
        break
        
if not text_col:
    print("⚠️ No standard text column found, selecting largest column")

    lengths = {}
    for col in df.columns:
        try:
            lengths[col] = df[col].astype(str).str.len().mean()
        except:
            continue

    text_col = max(lengths, key=lengths.get)

# Final fallback
if not title_col:
    title_col = df.columns[0]

print("✅ Title column:", title_col)
print("✅ Text column:", text_col)

author_col = next((c for c in df.columns if "author" in c), None)
date_col = next((c for c in df.columns if "date" in c or "year" in c), None)
category_col = next((c for c in df.columns if "category" in c), None)

df[text_col] = df[text_col].fillna("").astype(str)
df[title_col] = df[title_col].fillna("").astype(str)

df = df[df[text_col].str.strip() != ""]

if df.empty:
    print("⚠️ Dataset empty → using fallback data")

    df = pd.DataFrame({
        "title": ["Sample Research Paper"],
        "abstract": ["This paper discusses machine learning techniques."]
    })

    title_col = "title"
    text_col = "abstract"

print("📊 Final dataset size:", df.shape)

try:
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df[text_col])
except:
    raise Exception("❌ TF-IDF failed → text data issue")

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

    if not results:
        return []

    max_score = results[0][1] if results[0][1] != 0 else 1

    papers = []

    for i, score_val in results:
        row = df.iloc[i]

        title = str(row[title_col])
        summary = str(row[text_col])[:500] + "..."

        authors = row[author_col] if author_col else "Unknown"
        category = row[category_col] if category_col else "Research"

        year = "Unknown"
        if date_col and pd.notna(row[date_col]):
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
            "keywords": extract_keywords(summary),

            "drawbacks": extract_drawbacks(summary),
            "future_scope": future_scope(),

            "paper_link": f"https://arxiv.org/search/?query={search_title}&searchtype=all",
            "pdf_link": f"https://arxiv.org/search/?query={search_title}&searchtype=all",
            "scholar": f"https://scholar.google.com/scholar?q={search_title}"
        })

    return papers
