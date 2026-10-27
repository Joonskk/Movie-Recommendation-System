# Movie Recommendation System

[![tests](https://github.com/Joonskk/Movie-Recommendation-System/actions/workflows/test.yml/badge.svg)](https://github.com/Joonskk/Movie-Recommendation-System/actions/workflows/test.yml)

Item-to-item movie recommendations on the [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/) dataset: 32,000,204 ratings from 200,948 users on 84,432 movies. Pick a movie and it returns the movies whose ratings follow the same pattern.

The notebook goes from raw ratings to recommendations in four steps: explore the data, rank movies fairly, build a sparse user-item matrix, and find similar movies with cosine similarity.

## What's inside

| File | What it does |
| --- | --- |
| `main.ipynb` | Exploration, Bayesian-average ranking, the sparse matrix, and example recommendations |
| `recommender.py` | `create_X` builds the sparse matrix; `ItemRecommender` finds similar movies |
| `test_recommender.py` | pytest tests on a small hand-made dataset |

## How it works

**1. Exploration.** On average a user rated 159 movies and a movie got 379 ratings, but the spread is huge. *The Shawshank Redemption* has 102,929 ratings, while thousands of movies have one.

**2. Fair ranking with a Bayesian average.** Sorting by mean rating puts *Always a Bridesmaid (2000)* first, with a single 5-star rating. The Bayesian average pulls each movie's mean toward the global mean until it has enough ratings to stand on its own:

$$\text{Bayesian average} = \frac{C \cdot m + \sum x}{C + n}$$

where *C* is the average number of ratings per movie (379), *m* is the average movie rating (3.01), and *n* is the movie's own rating count. With it, *Shawshank*, *The Godfather* and *Planet Earth* come out on top.

**3. Sparse user-item matrix.** Ratings go into a 200,948 × 84,432 matrix with one row per user and one column per movie. Only 0.19% of the cells hold a rating, so it's stored as a SciPy CSR array, which keeps just the 32M ratings and not the 17 billion empty cells. Mapper dictionaries translate between MovieLens ids and matrix rows and columns.

**4. Item-to-item recommendations.** Each movie is a column of ratings. `ItemRecommender`:
- subtracts each user's mean rating, so a harsh rater's 3.0 and a generous rater's 4.5 both count as "liked it",
- finds the nearest movies by cosine similarity with scikit-learn's `NearestNeighbors`, directly on the sparse matrix,
- skips movies with fewer than 50 ratings, which would otherwise look identical to almost anything.

```python
from recommender import ItemRecommender, titles

rec = ItemRecommender(X, movie_mapper, movie_inv_mapper)
titles(rec.similar(movie_id=1, k=10), movies)   # movies similar to Toy Story (1995)
```

## Run it

1. Download [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/) and put `ratings.csv` and `movies.csv` in a `data/` folder. The dataset isn't in the repo because it's about 900 MB unzipped.
2. Install and open the notebook:

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
jupyter lab main.ipynb
```

Run the tests with `pytest`. They use a small hand-made dataset, so they don't need the download.

## Next steps

- Evaluate the recommendations offline: hold out each user's last ratings and measure precision@k.
- Compare with a latent-factor model (matrix factorization with truncated SVD or ALS).
