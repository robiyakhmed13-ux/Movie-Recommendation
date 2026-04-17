# =============================================================================
# Movie Recommendation System — Content-Based Filtering
# using TF-IDF Vectorization + Cosine Similarity
# Author: [Your Name]
# Dataset: TMDB 5000 Movie Dataset
#          https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
# =============================================================================
#
# This is a CONTENT-BASED recommendation system. It recommends movies that
# are most similar to a given movie based on textual features:
# genres, keywords, tagline, cast, and director.
#
# Pipeline:
#   Raw text features → combined string → TF-IDF matrix → Cosine Similarity
#   → sorted similarity scores → Top-N recommendations
# =============================================================================

import pandas as pd
import numpy as np
import difflib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# 1. Data Loading
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """Load the movies dataset and reset index for safe lookup."""
    df = pd.read_csv(filepath)

    # Ensure a clean integer index column for similarity lookups
    if 'index' not in df.columns:
        df = df.reset_index()

    print(f"Dataset loaded: {df.shape[0]} movies, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    return df


# =============================================================================
# 2. Exploratory Data Analysis
# =============================================================================

def plot_eda(df: pd.DataFrame) -> None:
    """
    Visualise the dataset:
      - Top 10 most common genres
      - Top 10 directors by movie count
      - Vote average distribution
      - Popularity distribution
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Movie Dataset — Exploratory Data Analysis", fontsize=17)

    # 1. Vote average distribution
    if 'vote_average' in df.columns:
        sns.histplot(df['vote_average'].dropna(), bins=25, kde=True,
                     ax=axes[0, 0], color='steelblue')
        axes[0, 0].set_title("Vote Average Distribution")
        axes[0, 0].set_xlabel("IMDb-style Vote Average")
        axes[0, 0].axvline(df['vote_average'].mean(), color='red',
                           linestyle='--', label=f"Mean: {df['vote_average'].mean():.1f}")
        axes[0, 0].legend()

    # 2. Popularity distribution (log scale)
    if 'popularity' in df.columns:
        pop = df['popularity'].dropna()
        axes[0, 1].hist(pop, bins=40, color='salmon', edgecolor='white', log=True)
        axes[0, 1].set_title("Popularity Distribution (log scale)")
        axes[0, 1].set_xlabel("Popularity Score")
        axes[0, 1].set_ylabel("Count (log)")

    # 3. Top 10 directors by movie count
    if 'director' in df.columns:
        top_directors = (
            df['director'].dropna()
            .value_counts()
            .head(10)
            .sort_values()
        )
        top_directors.plot(kind='barh', ax=axes[1, 0], color='steelblue')
        axes[1, 0].set_title("Top 10 Directors by Number of Films")
        axes[1, 0].set_xlabel("Number of Movies")

    # 4. Missing values overview
    selected = ['genres', 'keywords', 'tagline', 'cast', 'director']
    available = [c for c in selected if c in df.columns]
    missing_pct = (df[available].isnull().sum() / len(df) * 100).sort_values()
    missing_pct.plot(kind='barh', ax=axes[1, 1], color='lightcoral')
    axes[1, 1].set_title("Missing Values in Key Feature Columns (%)")
    axes[1, 1].set_xlabel("Missing (%)")
    axes[1, 1].axvline(0, color='black', lw=0.8)

    plt.tight_layout()
    plt.savefig("eda_plots.png", dpi=150)
    plt.show()
    print("EDA plots saved as 'eda_plots.png'")


# =============================================================================
# 3. Feature Engineering — Combine Text Features
# =============================================================================

def build_combined_features(df: pd.DataFrame) -> pd.Series:
    """
    Select the five most content-rich text columns and combine them
    into a single string per movie. Missing values are replaced with ''
    so they contribute nothing to the TF-IDF vector.

    Features used:
        genres, keywords, tagline, cast, director
    """
    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

    for feature in selected_features:
        if feature in df.columns:
            df[feature] = df[feature].fillna('')
        else:
            df[feature] = ''

    combined = (
        df['genres']   + ' ' +
        df['keywords'] + ' ' +
        df['tagline']  + ' ' +
        df['cast']     + ' ' +
        df['director']
    )
    print(f"Combined feature strings built for {len(combined)} movies.")
    return combined


# =============================================================================
# 4. TF-IDF Vectorisation
# =============================================================================

def vectorise(combined_features: pd.Series):
    """
    Transform combined text strings into a TF-IDF feature matrix.

    TF-IDF (Term Frequency–Inverse Document Frequency) down-weights
    words that appear in many movies (e.g. 'the', 'a') and up-weights
    distinctive words (e.g. a specific director's name or genre term).
    """
    vectorizer     = TfidfVectorizer()
    feature_matrix = vectorizer.fit_transform(combined_features)
    print(f"TF-IDF matrix shape: {feature_matrix.shape}")
    return vectorizer, feature_matrix


# =============================================================================
# 5. Compute Cosine Similarity
# =============================================================================

def compute_similarity(feature_matrix) -> np.ndarray:
    """
    Compute pairwise cosine similarity between all movie TF-IDF vectors.

    Cosine similarity measures the angle between two vectors:
      - 1.0  → identical content profile
      - 0.0  → completely dissimilar
    """
    similarity = cosine_similarity(feature_matrix)
    print(f"Similarity matrix shape: {similarity.shape}")
    return similarity


# =============================================================================
# 6. Recommend Movies
# =============================================================================

def recommend_movies(movie_name: str,
                     movies_df: pd.DataFrame,
                     similarity: np.ndarray,
                     top_n: int = 10) -> list:
    """
    Return the top-N most similar movies to the given title.

    Uses difflib.get_close_matches for fuzzy title matching —
    so typos like 'Avater' will still find 'Avatar'.

    Parameters
    ----------
    movie_name  : str — title entered by the user
    movies_df   : the full movies DataFrame
    similarity  : precomputed cosine similarity matrix
    top_n       : number of recommendations to return (default 10)

    Returns
    -------
    list of recommended movie titles
    """
    all_titles      = movies_df['title'].tolist()
    close_matches   = difflib.get_close_matches(movie_name, all_titles)

    if not close_matches:
        print(f"❌ No close match found for '{movie_name}'. "
              f"Please try a different title.")
        return []

    matched_title   = close_matches[0]
    print(f"\n🎬 Closest match found: '{matched_title}'")

    movie_index     = movies_df[movies_df['title'] == matched_title]['index'].values[0]
    similarity_scores = list(enumerate(similarity[movie_index]))
    sorted_movies   = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    print(f"\n🍿 Top {top_n} movies similar to '{matched_title}':\n")
    for rank, (idx, score) in enumerate(sorted_movies[1:top_n + 1], start=1):
        title = movies_df[movies_df['index'] == idx]['title'].values
        if len(title) > 0:
            recommendations.append((rank, title[0], round(score, 4)))
            print(f"  {rank:>2}. {title[0]:<45} (similarity: {score:.4f})")

    return recommendations


# =============================================================================
# 7. Visualise Recommendations
# =============================================================================

def plot_recommendations(recommendations: list, movie_name: str) -> None:
    """Bar chart of similarity scores for the top recommended movies."""
    if not recommendations:
        return

    ranks, titles, scores = zip(*recommendations)
    short_titles = [t[:35] + '…' if len(t) > 35 else t for t in titles]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(short_titles[::-1], scores[::-1], color='steelblue',
                    edgecolor='white')
    plt.xlabel("Cosine Similarity Score")
    plt.title(f"Top {len(recommendations)} Recommendations for  '{movie_name}'")
    for bar, score in zip(bars, scores[::-1]):
        plt.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                 f"{score:.4f}", va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig("recommendations.png", dpi=150)
    plt.show()
    print("Recommendations chart saved as 'recommendations.png'")


# =============================================================================
# Main Pipeline
# =============================================================================

if __name__ == "__main__":
    DATA_PATH = "movies.csv"   # update path if needed

    # 1. Load
    df = load_data(DATA_PATH)
    print("\nFirst 5 rows:\n", df[['title', 'genres', 'director']].head())

    # 2. EDA
    plot_eda(df)

    # 3. Feature Engineering
    combined_features = build_combined_features(df)

    # 4. TF-IDF
    vectorizer, feature_matrix = vectorise(combined_features)

    # 5. Cosine Similarity
    similarity = compute_similarity(feature_matrix)

    # 6 & 7. Recommend — try a few example movies
    for test_movie in ["Avatar", "The Dark Knight", "Inception"]:
        recs = recommend_movies(test_movie, df, similarity, top_n=10)
        if recs:
            plot_recommendations(recs, test_movie)

    # ── Interactive mode ──────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  🎬  INTERACTIVE MOVIE RECOMMENDER")
    print("="*55)
    user_movie = input("\nEnter your favourite movie name: ").strip()
    if user_movie:
        recs = recommend_movies(user_movie, df, similarity, top_n=10)
        if recs:
            plot_recommendations(recs, user_movie)
