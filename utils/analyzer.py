"""
=============================================================================
 ANALYZER MODULE - Keyword Extraction, Trend Analysis & Research Insights
=============================================================================
 This module provides utility functions for:
   - Extracting top keywords from research paper text using CountVectorizer
   - Determining research trend status based on publication year
   - Cleaning and formatting abstract text for display
   - Parsing author strings from dataset format to readable names
   - Safely parsing date values from various formats
=============================================================================
"""

import re
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer


# ──────────────────────────────────────────────────────────────────────────────
# KEYWORD EXTRACTION
# Uses CountVectorizer to find the most frequent meaningful terms in text.
# ──────────────────────────────────────────────────────────────────────────────
def extract_keywords(text, max_keywords=6):
    """
    Extract top keywords from a given text using CountVectorizer.
    
    Args:
        text (str): The input text (abstract/summary) to extract keywords from.
        max_keywords (int): Maximum number of keywords to return.
    
    Returns:
        list: A list of keyword strings extracted from the text.
    """
    try:
        # Skip if text is too short or empty
        if not text or len(text.strip()) < 20:
            return []
        
        # Use CountVectorizer with English stop words removed
        cv = CountVectorizer(
            stop_words="english",
            max_features=max_keywords,
            ngram_range=(1, 2),   # Include single words and bigrams
            min_df=1
        )
        cv.fit([text])
        keywords = list(cv.get_feature_names_out())
        return keywords
    except Exception:
        return []


# ──────────────────────────────────────────────────────────────────────────────
# TRENDING INDICATOR
# Classifies papers into trend categories based on their publication year.
# ──────────────────────────────────────────────────────────────────────────────
def get_trend_badge(year):
    """
    Return a trend indicator badge based on the publication year.
    
    Args:
        year (int): The publication year of the paper.
    
    Returns:
        tuple: (emoji_badge, label_text, css_class) for display purposes.
    """
    try:
        year = int(year)
    except (ValueError, TypeError):
        return ("📚", "Classic Research", "trend-classic")
    
    if year >= 2022:
        return ("🔥", "Hot & Latest", "trend-hot")
    elif year >= 2019:
        return ("📈", "Trending", "trend-trending")
    elif year >= 2015:
        return ("📊", "Established", "trend-established")
    elif year >= 2010:
        return ("📖", "Mature Research", "trend-mature")
    else:
        return ("📚", "Classic Foundation", "trend-classic")


# ──────────────────────────────────────────────────────────────────────────────
# ABSTRACT CLEANER
# Ensures abstracts display cleanly without cut words or broken formatting.
# ──────────────────────────────────────────────────────────────────────────────
def clean_abstract(text, max_chars=900):
    """
    Clean and truncate abstract text for proper display.
    Ensures text is not cut mid-word and preserves paragraph formatting.
    
    Args:
        text (str): Raw abstract text from the dataset.
        max_chars (int): Maximum characters to display (700-1000 range).
    
    Returns:
        str: Cleaned, properly truncated abstract text.
    """
    if not text or not isinstance(text, str):
        return "Abstract not available for this paper."
    
    # Remove excessive whitespace and normalize line breaks
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove LaTeX-style artifacts that are common in arXiv papers
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)  # \command{arg}
    text = text.replace('\\n', ' ')
    text = text.replace('\\', '')
    
    # If text is within limit, return as-is
    if len(text) <= max_chars:
        return text
    
    # Truncate at the last complete sentence within the character limit
    truncated = text[:max_chars]
    
    # Try to find the last sentence-ending punctuation
    last_period = truncated.rfind('.')
    last_question = truncated.rfind('?')
    last_exclaim = truncated.rfind('!')
    
    # Use the latest sentence boundary found
    best_cut = max(last_period, last_question, last_exclaim)
    
    if best_cut > max_chars * 0.5:  # Only cut at sentence if it's past halfway
        return truncated[:best_cut + 1]
    
    # Fallback: cut at last space to avoid breaking words
    last_space = truncated.rfind(' ')
    if last_space > 0:
        return truncated[:last_space] + "..."
    
    return truncated + "..."


# ──────────────────────────────────────────────────────────────────────────────
# AUTHOR PARSER
# Converts dataset author format into clean, readable author names.
# ──────────────────────────────────────────────────────────────────────────────
def parse_authors(author_string):
    """
    Parse author string from dataset format to clean readable format.
    Handles formats like: "['Author A', 'Author B']" or "Author A, Author B"
    
    Args:
        author_string: Raw author data from the dataset.
    
    Returns:
        str: Cleaned, comma-separated author names.
    """
    if not author_string or str(author_string).strip() in ['', 'nan', 'None', '[]']:
        return "Author information not available"
    
    text = str(author_string).strip()
    
    # Handle Python list-like format: ['Author A', 'Author B']
    if text.startswith('[') and text.endswith(']'):
        # Remove brackets
        text = text[1:-1]
        # Split by comma and clean each author name
        authors = []
        for part in text.split(','):
            name = part.strip().strip("'").strip('"').strip()
            if name and name not in ['', 'nan', 'None']:
                authors.append(name)
        if authors:
            return ", ".join(authors)
    
    # Handle comma-separated format
    if ',' in text:
        authors = [a.strip() for a in text.split(',') if a.strip()]
        return ", ".join(authors)
    
    return text if text else "Author information not available"


# ──────────────────────────────────────────────────────────────────────────────
# DATE PARSER
# Safely parses dates from various formats found in research datasets.
# ──────────────────────────────────────────────────────────────────────────────
def parse_year(date_value):
    """
    Safely extract the year from various date formats.
    Handles: '8/1/93', '2020-01-15', '2020', datetime objects, etc.
    
    Args:
        date_value: Raw date value from the dataset.
    
    Returns:
        int or None: Extracted year, or None if parsing fails.
    """
    if date_value is None or str(date_value).strip() in ['', 'nan', 'None', 'NaT']:
        return None
    
    # If already a datetime object
    if isinstance(date_value, datetime):
        return date_value.year
    
    text = str(date_value).strip()
    
    # Try direct year (e.g., "2020")
    if re.match(r'^\d{4}$', text):
        return int(text)
    
    # Try common date formats
    date_formats = [
        '%m/%d/%y',      # 8/1/93
        '%m/%d/%Y',      # 8/1/1993
        '%Y-%m-%d',      # 2020-01-15
        '%Y/%m/%d',      # 2020/01/15
        '%d-%m-%Y',      # 15-01-2020
        '%d/%m/%Y',      # 15/01/2020
        '%B %d, %Y',     # January 15, 2020
        '%b %d, %Y',     # Jan 15, 2020
        '%Y-%m-%dT%H:%M:%S',  # ISO format
    ]
    
    for fmt in date_formats:
        try:
            dt = datetime.strptime(text, fmt)
            year = dt.year
            # Fix 2-digit year interpretation (93 -> 1993 not 2093)
            if year > 2030:
                year -= 100
            return year
        except ValueError:
            continue
    
    # Last resort: try to find a 4-digit year in the string
    year_match = re.search(r'(19|20)\d{2}', text)
    if year_match:
        return int(year_match.group())
    
    return None


# ──────────────────────────────────────────────────────────────────────────────
# DOMAIN / CATEGORY MAPPER
# Maps arXiv category codes and names to user-friendly domain labels.
# ──────────────────────────────────────────────────────────────────────────────

# Mapping of user-friendly domain names to keywords found in arXiv categories
DOMAIN_MAP = {
    "All": [],
    "AI": ["artificial intelligence", "cs.ai"],
    "ML": ["machine learning", "cs.lg", "stat.ml"],
    "NLP": ["natural language", "computation and language", "cs.cl"],
    "Computer Vision": ["computer vision", "pattern recognition", "cs.cv"],
    "Security": ["cryptography", "security", "cs.cr"],
    "Robotics": ["robotics", "cs.ro"],
    "Neural Networks": ["neural", "evolutionary", "cs.ne"],
    "Data Science": ["information retrieval", "databases", "statistics", "cs.ir", "cs.db"],
    "Cloud / Distributed": ["distributed", "parallel", "cluster", "cs.dc"],
    "Signal Processing": ["signal processing", "eess.sp", "image and video"],
    "Quantum": ["quantum", "quant-ph"],
    "Software": ["software engineering", "cs.se"],
    "Medical / Bio": ["medical", "bio", "health", "neurons and cognition", "q-bio"],
    "Finance": ["finance", "econom", "q-fin"],
    "IoT / Networks": ["networking", "internet", "iot", "cs.ni"],
}


def get_domain_list():
    """Return the list of available domain filter names."""
    return list(DOMAIN_MAP.keys())


def filter_by_domain(df, domain, category_col):
    """
    Filter DataFrame rows by the selected research domain.
    
    Args:
        df: pandas DataFrame of papers.
        domain (str): Selected domain name from DOMAIN_MAP.
        category_col (str): Name of the category column in the DataFrame.
    
    Returns:
        pandas DataFrame: Filtered subset of papers matching the domain.
    """
    if domain == "All" or not category_col:
        return df
    
    keywords = DOMAIN_MAP.get(domain, [])
    if not keywords:
        return df
    
    # Build a mask: row matches if its category contains ANY of the domain keywords
    mask = df[category_col].fillna("").str.lower().apply(
        lambda cat: any(kw in cat for kw in keywords)
    )
    
    # Also check category_code column if it exists
    if "category_code" in df.columns:
        code_mask = df["category_code"].fillna("").str.lower().apply(
            lambda code: any(kw in code for kw in keywords)
        )
        mask = mask | code_mask
    
    filtered = df[mask]
    
    # If filter returns too few results, return original
    if len(filtered) < 10:
        return df
    
    return filtered