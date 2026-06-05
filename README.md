# 🎬 CineMatch — Movie Recommendation Engine

A **content-based movie recommendation system** built with Python. It uses **TF-IDF vectorization** and **cosine similarity** to find movies similar to any title you provide, based on genres, cast, keywords, and plot overview.

---

## How It Works

1. **Feature Engineering** — For each movie, genres, cast, keywords, and overview are combined into a single text "soup". Genres and keywords are repeated to give them higher weight.
2. **TF-IDF Vectorization** — The feature soup is converted into numerical vectors using `TfidfVectorizer` (removes stop words, caps vocabulary at 5000 terms).
3. **Cosine Similarity** — A pairwise similarity matrix is computed across all movies. Movies with a higher cosine similarity score are more alike.
4. **Top-N Recommendations** — Given a query movie, the engine ranks all others by similarity score and returns the top N matches above a configurable threshold.
5. **Evaluation** — Recommendation quality is measured using **Precision@K** (fraction of top-K recommendations sharing at least one genre with the query movie).

---

## Project Structure

```
CineMatch/
│
├── main.py           # Entry point — CLI + interactive mode
├── recommender.py    # Core logic: data loading, feature engineering, similarity, evaluation
├── requirements.txt  # Dependencies
│
└── data/
    └── movies.csv    # Movie dataset (100 movies, extendable with MovieLens)
```

---

## Setup

```bash
git clone https://github.com/your-username/CineMatch.git
cd CineMatch
pip install -r requirements.txt
```

---

## Usage

### Interactive mode (default)
```bash
python main.py
```
```
Enter movie title: Inception
How many recommendations? (default 5): 5

  Finding recommendations for: 'Inception'

   title                                  year  genres                    similarity_score
1  Interstellar                           2014  Adventure Drama Sci-Fi    0.4821
2  The Matrix                             1999  Action Sci-Fi             0.4103
3  Arrival                                2016  Drama Mystery Sci-Fi      0.3892
4  Blade Runner 2049                      2017  Action Drama Sci-Fi       0.3441
5  Ex Machina                             2014  Drama Sci-Fi Thriller     0.2987
```

### Single query via command line
```bash
python main.py --movie "The Godfather" --top 5
python main.py --movie "Toy Story" --top 10 --threshold 0.1
```

### Exploratory Data Analysis
```bash
python main.py --eda
```

### Precision@K Evaluation
```bash
python main.py --eval
```

---

## Sample Output

```
  CineMatch — Movie Recommendation Engine

[1/4] Loading data...
  Loaded 100 movies.
[2/4] Engineering features...
[3/4] Building TF-IDF matrix...
  TF-IDF matrix shape: (100, 1847)
[4/4] Computing cosine similarity matrix...
  Similarity matrix shape: (100, 100)

  Finding recommendations for: 'The Dark Knight'

   title                    year  genres               similarity_score
1  Batman Begins            2005  Action Crime Drama   0.5102
2  The Avengers             2012  Action Adventure     0.3874
3  Se7en                    1995  Crime Drama Mystery  0.3201
4  The Departed             2006  Crime Drama Thriller 0.2954
5  No Country for Old Men   2007  Crime Drama Thriller 0.2711

  Precision@5 (genre match): 1.00
```

---

## Dataset

The project ships with a curated dataset of 100 well-known movies. To scale up, replace `data/movies.csv` with the [MovieLens dataset](https://grouplens.org/datasets/movielens/) (same column format).

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.8+ | Core language |
| Pandas | Data loading, cleaning, EDA |
| NumPy | Numerical operations |
| Scikit-learn | TF-IDF vectorization, cosine similarity |

---

## Key Concepts

- **TF-IDF (Term Frequency–Inverse Document Frequency)** — Gives higher weight to words that are distinctive to a movie and lower weight to common words.
- **Cosine Similarity** — Measures the angle between two feature vectors. A score of 1.0 means identical; 0.0 means no overlap.
- **Precision@K** — A standard information retrieval metric. Precision@5 = 0.8 means 4 out of 5 recommendations are relevant (share a genre).
- **Content-Based Filtering** — Recommends movies similar to what the user already likes, without needing other users' data.
