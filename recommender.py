"""
CineMatch - Movie Recommendation Engine
========================================
Content-based filtering using TF-IDF vectorization and cosine similarity.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os


# ─────────────────────────────────────────────
#  Data Loading & Cleaning
# ─────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load movies CSV and perform basic cleaning."""
    df = pd.read_csv(filepath)

    print(f"  Loaded {len(df)} movies.")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Missing values:\n{df.isnull().sum()}\n")

    # Fill missing text fields with empty string
    for col in ["genres", "cast", "keywords", "overview"]:
        df[col] = df[col].fillna("")

    # Normalize title for easier lookup
    df["title_clean"] = df["title"].str.lower().str.strip()

    return df


# ─────────────────────────────────────────────
#  Feature Engineering
# ─────────────────────────────────────────────

def build_feature_soup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine genres, cast, keywords, and overview into a single
    'soup' string for each movie. This is what TF-IDF will vectorize.

    Genres and keywords are weighted more by repeating them.
    """
    def make_soup(row):
        genres   = row["genres"].replace(" ", "").replace(",", " ")
        cast     = row["cast"].replace(",", " ")
        keywords = row["keywords"].replace(",", " ")
        overview = row["overview"]

        # Repeat genres/keywords to give them more weight
        return f"{genres} {genres} {keywords} {keywords} {cast} {overview}"

    df["soup"] = df.apply(make_soup, axis=1)
    return df


def compute_tfidf_matrix(df: pd.DataFrame):
    """
    Build TF-IDF matrix from the feature soup.
    - stop_words='english'  removes common words like 'the', 'a', 'in'
    - max_features caps vocabulary size for performance
    """
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df["soup"])

    print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"  Vocabulary size: {len(tfidf.vocabulary_)}\n")

    return tfidf_matrix, tfidf


def compute_similarity_matrix(tfidf_matrix) -> np.ndarray:
    """Compute pairwise cosine similarity between all movies."""
    similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print(f"  Similarity matrix shape: {similarity.shape}\n")
    return similarity


# ─────────────────────────────────────────────
#  Recommendation Logic
# ─────────────────────────────────────────────

def get_recommendations(
    title: str,
    df: pd.DataFrame,
    similarity_matrix: np.ndarray,
    top_n: int = 5,
    similarity_threshold: float = 0.05
) -> pd.DataFrame:
    """
    Return top-N movie recommendations for a given title.

    Parameters
    ----------
    title               : Movie title to search for
    df                  : Cleaned movie DataFrame
    similarity_matrix   : Precomputed cosine similarity matrix
    top_n               : Number of recommendations to return
    similarity_threshold: Minimum similarity score to include
    """
    title_lower = title.lower().strip()

    # Fuzzy title match (substring)
    matches = df[df["title_clean"].str.contains(title_lower, na=False)]

    if matches.empty:
        print(f"  Movie '{title}' not found. Try a different title.\n")
        return pd.DataFrame()

    # Use the first match if multiple
    movie_idx = matches.index[0]
    matched_title = df.loc[movie_idx, "title"]

    if len(matches) > 1:
        print(f"  Multiple matches found. Using: '{matched_title}'\n")
    else:
        print(f"  Finding recommendations for: '{matched_title}'\n")

    # Get similarity scores for this movie against all others
    sim_scores = list(enumerate(similarity_matrix[movie_idx]))

    # Sort by similarity descending, exclude the movie itself (idx == movie_idx)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [(idx, score) for idx, score in sim_scores
                  if idx != movie_idx and score >= similarity_threshold]

    # Take top N
    top_scores = sim_scores[:top_n]

    if not top_scores:
        print("  No similar movies found above the similarity threshold.\n")
        return pd.DataFrame()

    indices = [i[0] for i in top_scores]
    scores  = [round(i[1], 4) for i in top_scores]

    results = df.loc[indices, ["title", "year", "genres"]].copy()
    results["similarity_score"] = scores
    results = results.reset_index(drop=True)
    results.index += 1  # Start ranking from 1

    return results


# ─────────────────────────────────────────────
#  Evaluation — Precision@K
# ─────────────────────────────────────────────

def precision_at_k(
    title: str,
    df: pd.DataFrame,
    similarity_matrix: np.ndarray,
    k: int = 5,
    relevant_genre: str = None
) -> float:
    """
    Compute Precision@K: what fraction of the top-K recommendations
    share at least one genre with the query movie.

    If relevant_genre is None, the query movie's first genre is used.
    """
    title_lower = title.lower().strip()
    matches = df[df["title_clean"].str.contains(title_lower, na=False)]

    if matches.empty:
        print(f"  Movie '{title}' not found for evaluation.\n")
        return 0.0

    movie_idx = matches.index[0]
    query_genres = set(df.loc[movie_idx, "genres"].lower().split())

    if relevant_genre:
        query_genres = {relevant_genre.lower()}

    recs = get_recommendations(title, df, similarity_matrix, top_n=k)

    if recs.empty:
        return 0.0

    hits = 0
    for _, row in recs.iterrows():
        rec_genres = set(row["genres"].lower().split())
        if query_genres & rec_genres:  # intersection
            hits += 1

    precision = hits / k
    return precision


# ─────────────────────────────────────────────
#  EDA Helper
# ─────────────────────────────────────────────

def run_eda(df: pd.DataFrame):
    """Print basic exploratory data analysis stats."""
    print("=" * 50)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 50)

    print(f"\nTotal movies    : {len(df)}")
    print(f"Year range      : {df['year'].min()} – {df['year'].max()}")

    # Genre frequency
    all_genres = df["genres"].str.split().explode()
    top_genres = all_genres.value_counts().head(10)
    print(f"\nTop 10 genres:")
    for genre, count in top_genres.items():
        print(f"  {genre:<20} {count}")

    # Overview length distribution
    df["overview_len"] = df["overview"].str.split().str.len()
    print(f"\nOverview word count — mean: {df['overview_len'].mean():.1f}, "
          f"min: {df['overview_len'].min()}, max: {df['overview_len'].max()}")
    print()
