"""
CineMatch - Movie Recommendation Engine
========================================
Run this file to get movie recommendations from the command line.

Usage:
    python main.py
    python main.py --movie "Inception" --top 5
    python main.py --movie "The Matrix" --top 10 --threshold 0.1
    python main.py --eda
"""

import argparse
import os
from recommender import (
    load_data,
    build_feature_soup,
    compute_tfidf_matrix,
    compute_similarity_matrix,
    get_recommendations,
    precision_at_k,
    run_eda,
)


DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "movies.csv")


def build_engine(verbose=True):
    """Load data and build the recommendation engine."""
    if verbose:
        print("\n" + "=" * 50)
        print("  CineMatch — Movie Recommendation Engine")
        print("=" * 50)
        print("\n[1/4] Loading data...")

    df = load_data(DATA_PATH)

    if verbose:
        print("[2/4] Engineering features...")
    df = build_feature_soup(df)

    if verbose:
        print("[3/4] Building TF-IDF matrix...")
    tfidf_matrix, _ = compute_tfidf_matrix(df)

    if verbose:
        print("[4/4] Computing cosine similarity matrix...")
    similarity_matrix = compute_similarity_matrix(tfidf_matrix)

    if verbose:
        print("  Engine ready.\n")

    return df, similarity_matrix


def interactive_mode(df, similarity_matrix):
    """Simple interactive CLI loop."""
    print("─" * 50)
    print("  Type a movie title to get recommendations.")
    print("  Type 'quit' or 'exit' to stop.")
    print("─" * 50 + "\n")

    while True:
        user_input = input("Enter movie title: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("\nGoodbye! Happy watching.")
            break

        try:
            top_n = int(input("How many recommendations? (default 5): ").strip() or 5)
        except ValueError:
            top_n = 5

        print()
        recs = get_recommendations(
            title=user_input,
            df=df,
            similarity_matrix=similarity_matrix,
            top_n=top_n,
        )

        if not recs.empty:
            print(recs.to_string())
            print()

            # Show Precision@K
            p_at_k = precision_at_k(user_input, df, similarity_matrix, k=top_n)
            print(f"  Precision@{top_n} (genre match): {p_at_k:.2f}")

        print()


def main():
    parser = argparse.ArgumentParser(
        description="CineMatch — Content-Based Movie Recommendation Engine"
    )
    parser.add_argument(
        "--movie", "-m",
        type=str,
        default=None,
        help="Movie title to get recommendations for"
    )
    parser.add_argument(
        "--top", "-n",
        type=int,
        default=5,
        help="Number of recommendations (default: 5)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.05,
        help="Minimum similarity score threshold (default: 0.05)"
    )
    parser.add_argument(
        "--eda",
        action="store_true",
        help="Run exploratory data analysis and exit"
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run Precision@K evaluation on sample movies and exit"
    )

    args = parser.parse_args()

    df, similarity_matrix = build_engine(verbose=True)

    # ── EDA mode ──────────────────────────────
    if args.eda:
        run_eda(df)
        return

    # ── Evaluation mode ───────────────────────
    if args.eval:
        print("=" * 50)
        print("PRECISION@K EVALUATION")
        print("=" * 50)
        test_movies = [
            "Inception",
            "The Matrix",
            "The Godfather",
            "Toy Story",
            "The Shawshank Redemption",
        ]
        k = 5
        scores = []
        for movie in test_movies:
            p = precision_at_k(movie, df, similarity_matrix, k=k)
            scores.append(p)
            print(f"  {movie:<40} Precision@{k}: {p:.2f}")
        avg = sum(scores) / len(scores)
        print(f"\n  Average Precision@{k}: {avg:.2f}")
        return

    # ── Single query mode ─────────────────────
    if args.movie:
        recs = get_recommendations(
            title=args.movie,
            df=df,
            similarity_matrix=similarity_matrix,
            top_n=args.top,
            similarity_threshold=args.threshold,
        )
        if not recs.empty:
            print(recs.to_string())
            print()
            p = precision_at_k(args.movie, df, similarity_matrix, k=args.top)
            print(f"  Precision@{args.top} (genre match): {p:.2f}\n")
        return

    # ── Interactive mode (default) ────────────
    interactive_mode(df, similarity_matrix)


if __name__ == "__main__":
    main()
