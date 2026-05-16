"""
=============================================================================
 RECOMMENDER ENGINE - Smart Research Paper Recommendation System
=============================================================================
 Core ML logic using SIMPLE techniques only:
   - TF-IDF Vectorizer for text feature extraction
   - Cosine Similarity for relevance matching
   - Recency-based scoring to prioritize newer papers
 
 Scoring Formula:
   final_score = (0.75 * cosine_similarity) + (0.25 * recency_score)
 
 No deep learning, no transformers, no external APIs.
 Beginner-friendly and easy to explain in a viva.
=============================================================================
"""

import os
import re
import urllib.parse

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.analyzer import (
    parse_authors,
    parse_year,
    clean_abstract,
    extract_keywords,
    get_trend_badge,
    filter_by_domain,
)


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

# Maximum rows to load from dataset (for performance optimization)
MAX_ROWS = 50000

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to the dataset file
DATA_DIR = os.path.join(BASE_DIR, "dataset")


# ──────────────────────────────────────────────────────────────────────────────
# SMART COLUMN DETECTION
# Automatically detects the correct column names regardless of dataset format.
# ──────────────────────────────────────────────────────────────────────────────
def detect_column(df, candidates):
    """
    Detect the correct column name from a list of candidate names.
    
    Args:
        df: pandas DataFrame to search in.
        candidates (list): List of possible column names to check.
    
    Returns:
        str or None: The first matching column name found, or None.
    """
    columns_lower = {col.lower().strip(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in columns_lower:
            return columns_lower[candidate.lower()]
    return None


# ──────────────────────────────────────────────────────────────────────────────
# DATASET LOADING & PREPROCESSING
# Loads the CSV dataset with robust error handling and caching.
# ──────────────────────────────────────────────────────────────────────────────
def find_dataset_file():
    """
    Find the CSV dataset file in the dataset directory.
    Handles spaces in filenames and multiple CSV files.
    
    Returns:
        str: Full path to the dataset CSV file.
    
    Raises:
        FileNotFoundError: If no CSV file is found in the dataset directory.
    """
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")
    
    # Look for CSV files in the dataset directory
    csv_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.csv')]
    
    if not csv_files:
        raise FileNotFoundError("No CSV dataset file found in the 'dataset' folder.")
    
    # Prefer sampled file (for deployment), then any arxiv file
    sampled = [f for f in csv_files if 'sampled' in f.lower()]
    if sampled:
        chosen = sampled[0]
    else:
        arxiv_files = [f for f in csv_files if 'arxiv' in f.lower()]
        chosen = arxiv_files[0] if arxiv_files else csv_files[0]
    
    return os.path.join(DATA_DIR, chosen)


def load_and_prepare_data():
    """
    Load the dataset CSV file and prepare it for TF-IDF processing.
    
    This function:
      1. Finds and loads the CSV file with error handling
      2. Detects column names automatically
      3. Cleans and preprocesses text data
      4. Parses author names and dates
      5. Builds the TF-IDF matrix for similarity computation
    
    Returns:
        tuple: (df, tfidf_matrix, vectorizer, column_info_dict)
    """
    # ── Step 1: Find and load the dataset ──
    data_path = find_dataset_file()
    
    try:
        # Try reading with different encodings for robustness
        df = pd.read_csv(
            data_path,
            nrows=MAX_ROWS,
            encoding='utf-8',
            on_bad_lines='skip',
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            data_path,
            nrows=MAX_ROWS,
            encoding='latin-1',
            on_bad_lines='skip',
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {str(e)}")
    
    if df.empty:
        raise RuntimeError("Dataset is empty after loading.")
    
    # ── Step 2: Detect columns automatically ──
    title_col = detect_column(df, ['title', 'titles', 'paper_title', 'name'])
    text_col = detect_column(df, ['summary', 'abstract', 'description', 'text', 'content'])
    author_col = detect_column(df, ['authors', 'author', 'creator', 'writer', 'first_author'])
    date_col = detect_column(df, ['published_date', 'published', 'update_date', 'updated_date',
                                   'publish_date', 'date', 'year', 'publication_date'])
    category_col = detect_column(df, ['category', 'categories', 'subject', 'domain', 'field',
                                       'category_code', 'topic'])
    id_col = detect_column(df, ['id', 'paper_id', 'arxiv_id', 'doi'])
    
    if not title_col:
        raise RuntimeError("Could not find a 'title' column in the dataset.")
    if not text_col:
        raise RuntimeError("Could not find an 'abstract' or 'summary' column in the dataset.")
    
    # ── Step 3: Clean and preprocess data ──
    # Fill missing values to prevent NaN errors
    df[title_col] = df[title_col].fillna("").astype(str).str.replace('\n', ' ', regex=False).str.replace('\r', ' ', regex=False).str.strip()
    df[text_col] = df[text_col].fillna("").astype(str)
    
    if author_col:
        df[author_col] = df[author_col].fillna("").astype(str)
    if category_col:
        df[category_col] = df[category_col].fillna("").astype(str)
    
    # Remove rows with empty titles or abstracts
    df = df[df[title_col].str.strip() != ""]
    df = df[df[text_col].str.strip() != ""]
    df = df.reset_index(drop=True)
    
    if len(df) == 0:
        raise RuntimeError("No valid papers found after cleaning.")
    
    # ── Step 4: Parse years for all rows (used in recency scoring) ──
    if date_col:
        df['_parsed_year'] = df[date_col].apply(parse_year)
    else:
        df['_parsed_year'] = None
    
    # ── Step 5: Build TF-IDF matrix ──
    # Combine title and abstract for better matching
    df['_search_text'] = df[title_col] + " " + df[text_col]
    
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=20000,       # Limit features for performance
        sublinear_tf=True,        # Apply log normalization (reduces impact of very frequent terms)
        ngram_range=(1, 2),       # Include bigrams for better matching
        min_df=2,                 # Ignore very rare terms
        max_df=0.95,              # Ignore terms appearing in >95% of docs
    )
    
    tfidf_matrix = vectorizer.fit_transform(df['_search_text'])
    
    # ── Store column info for later use ──
    col_info = {
        'title': title_col,
        'text': text_col,
        'author': author_col,
        'date': date_col,
        'category': category_col,
        'id': id_col,
    }
    
    return df, tfidf_matrix, vectorizer, col_info


# ──────────────────────────────────────────────────────────────────────────────
# LINK GENERATION
# Generates real arXiv, PDF, and Google Scholar URLs for each paper.
# ──────────────────────────────────────────────────────────────────────────────
def generate_paper_links(paper_id, title):
    """
    Generate real, clickable links for a research paper.
    
    Args:
        paper_id (str): The arXiv paper ID (e.g., 'cs-9308101v1' or '2001.12345v1').
        title (str): The paper title for Google Scholar search.
    
    Returns:
        dict: Dictionary with 'arxiv', 'pdf', and 'scholar' URLs.
    """
    # URL-encode the title for search links
    encoded_title = urllib.parse.quote_plus(str(title))
    
    # Default links using title search
    links = {
        'arxiv': f"https://arxiv.org/search/?query={encoded_title}&searchtype=all",
        'pdf': f"https://arxiv.org/search/?query={encoded_title}&searchtype=all",
        'scholar': f"https://scholar.google.com/scholar?q={encoded_title}",
    }
    
    # If we have a real arXiv ID, generate direct links
    if paper_id and str(paper_id).strip() not in ['', 'nan', 'None']:
        clean_id = str(paper_id).strip()
        
        # Remove version suffix (v1, v2, etc.)
        clean_id = re.sub(r'v\d+$', '', clean_id)
        
        # Handle IDs starting with 'abs-' (e.g. abs-1301.3524v1 → 1301.3524)
        if clean_id.startswith('abs-'):
            arxiv_id = clean_id[4:]  # Strip 'abs-' prefix
        elif re.match(r'^[a-z]+-\d+', clean_id, re.IGNORECASE):
            # Old format: cs-9308101 → cs/9308101
            arxiv_id = clean_id.replace('-', '/', 1)
        elif re.match(r'^\d{4}\.\d+', clean_id):
            # New format: 2001.12345
            arxiv_id = clean_id
        else:
            arxiv_id = clean_id
        
        links['arxiv'] = f"https://arxiv.org/abs/{arxiv_id}"
        links['pdf'] = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    
    return links


# ──────────────────────────────────────────────────────────────────────────────
# MAIN RECOMMENDATION FUNCTION
# Core logic: TF-IDF + Cosine Similarity + Recency Scoring
# ──────────────────────────────────────────────────────────────────────────────
def recommend_papers(query, df, tfidf_matrix, vectorizer, col_info,
                     domain="All", top_n=10):
    """
    Recommend research papers based on a user query using ML techniques.
    
    Algorithm:
      1. Transform user query using the fitted TF-IDF vectorizer
      2. Compute cosine similarity between query and all paper vectors
      3. Apply domain filtering if selected
      4. Calculate recency score for each paper
      5. Combine scores: final = 0.75 * similarity + 0.25 * recency
      6. Return top-N papers sorted by final score
    
    Args:
        query (str): User's search query (e.g., "machine learning").
        df: pandas DataFrame of all papers.
        tfidf_matrix: Pre-computed TF-IDF sparse matrix.
        vectorizer: Fitted TfidfVectorizer instance.
        col_info (dict): Column name mapping.
        domain (str): Selected domain filter (default: "All").
        top_n (int): Number of top papers to return (default: 10).
    
    Returns:
        list: List of dictionaries, each containing paper details and scores.
    """
    if not query or not query.strip():
        return []
    
    # ── Step 1: Apply domain filter to narrow search space ──
    category_col = col_info['category']
    if domain != "All" and category_col:
        filtered_df = filter_by_domain(df, domain, category_col)
        # Get indices of filtered rows in the original DataFrame
        filtered_indices = filtered_df.index.tolist()
    else:
        filtered_indices = list(range(len(df)))
    
    if len(filtered_indices) == 0:
        return []
    
    # ── Step 2: Transform query to TF-IDF vector ──
    try:
        query_vec = vectorizer.transform([query])
    except Exception:
        return []
    
    # ── Step 3: Compute cosine similarity ──
    # Only compute for filtered indices for efficiency
    from scipy.sparse import vstack as sparse_vstack
    
    filtered_matrix = tfidf_matrix[filtered_indices]
    similarities = cosine_similarity(query_vec, filtered_matrix).flatten()
    
    # ── Step 4: Get top candidates (take more than needed for scoring) ──
    candidate_count = min(200, len(similarities))
    top_local_indices = similarities.argsort()[-candidate_count:][::-1]
    
    # ── Step 5: Calculate final scores with recency weighting ──
    current_year = 2026
    min_year = 1990
    year_range = current_year - min_year
    
    scored_results = []
    
    for local_idx in top_local_indices:
        global_idx = filtered_indices[local_idx]
        sim_score = float(similarities[local_idx])
        
        # Skip papers with zero similarity
        if sim_score <= 0.0:
            continue
        
        # Calculate recency score (0.0 to 1.0, higher = more recent)
        # Steep exponential decay so 2024-2025 papers rank MUCH higher
        year = df.iloc[global_idx].get('_parsed_year', None)
        year_int = 2000
        try:
            if year is not None and pd.notna(year):
                year_int = int(year)
                years_ago = max(0, current_year - year_int)
                recency_score = max(0.02, 1.0 / (1.0 + 0.2 * years_ago ** 1.3))
            else:
                recency_score = 0.02
        except (ValueError, TypeError):
            recency_score = 0.02
        
        # ── SCORING FORMULA (Tiered Recency) ──
        # Hard penalty tiers ensure latest papers always rank at top
        if year_int >= 2023:
            recency_multiplier = 1.0     # Latest: full score
        elif year_int >= 2020:
            recency_multiplier = 0.80    # Recent: slight penalty
        elif year_int >= 2017:
            recency_multiplier = 0.55    # Older: moderate penalty
        elif year_int >= 2014:
            recency_multiplier = 0.40    # Old: heavy penalty
        else:
            recency_multiplier = 0.25    # Very old: severe penalty
        
        final_score = sim_score * recency_multiplier
        
        scored_results.append((global_idx, final_score, sim_score, recency_score, year_int))
    
    # Sort by year (latest first) to ensure papers close to 2026 are at the top, then by final score
    scored_results.sort(key=lambda x: (x[4], x[1]), reverse=True)
    
    # Take only top_n results
    scored_results = scored_results[:top_n]
    
    if not scored_results:
        return []
    
    # ── Step 6: Build result dictionaries ──
    max_final = scored_results[0][1] if scored_results[0][1] > 0 else 1.0
    
    papers = []
    
    for rank, (idx, final_sc, sim_sc, rec_sc, _yr) in enumerate(scored_results, 1):
        row = df.iloc[idx]
        
        # Extract paper details
        title = str(row[col_info['title']]).strip()
        raw_abstract = str(row[col_info['text']])
        
        # Parse authors properly
        authors = "Author information not available"
        if col_info['author']:
            authors = parse_authors(row[col_info['author']])
        
        # Get year (handle NaN carefully)
        year = row.get('_parsed_year', None)
        try:
            if year is not None and pd.notna(year):
                year_display = str(int(year))
            else:
                year_display = "Year N/A"
        except (ValueError, TypeError):
            year_display = "Year N/A"
        
        # Get category
        category = ""
        if col_info['category']:
            category = str(row[col_info['category']]).strip()
        
        # Get paper ID for link generation
        paper_id = ""
        if col_info['id']:
            paper_id = str(row[col_info['id']])
        
        # Clean abstract for display (700-1000 characters)
        abstract = clean_abstract(raw_abstract, max_chars=900)
        
        # Extract keywords from the abstract
        keywords = extract_keywords(raw_abstract)
        
        # Get trend badge
        trend_emoji, trend_label, trend_class = get_trend_badge(year)
        
        # Generate real links
        links = generate_paper_links(paper_id, title)
        
        # Calculate display score as percentage (0-100)
        relevance_pct = round((final_sc / max_final) * 100, 1)
        similarity_pct = round(sim_sc * 100, 1)
        recency_pct = round(rec_sc * 100, 1)
        
        # Extract drawbacks from abstract text
        abs_lower = raw_abstract.lower()
        if any(w in abs_lower for w in ['however', 'limitation', 'drawback', 'challenge', 'but ', 'although']):
            drawbacks = "The paper discusses certain limitations or challenges in its approach."
        else:
            drawbacks = "Specific limitations are not explicitly mentioned in the abstract."
        
        # Future scope based on category
        future_scope = f"This research in {category if category else 'this domain'} can be extended using larger datasets, real-time systems, and advanced optimization techniques."
        
        papers.append({
            'rank': rank,
            'title': title,
            'authors': authors,
            'year': year_display,
            'category': category,
            'abstract': abstract,
            'keywords': keywords,
            'relevance_score': relevance_pct,
            'similarity_score': similarity_pct,
            'recency_score': recency_pct,
            'trend_emoji': trend_emoji,
            'trend_label': trend_label,
            'trend_class': trend_class,
            'arxiv_link': links['arxiv'],
            'pdf_link': links['pdf'],
            'scholar_link': links['scholar'],
            'is_top': rank <= 3,
            'drawbacks': drawbacks,
            'future_scope': future_scope,
        })
    
    return papers