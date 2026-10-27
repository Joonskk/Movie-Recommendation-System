"""Item-to-item movie recommendations from a sparse user-item rating matrix.

Two movies are similar when the same users rated them in a similar way.
Each movie is a column of the user-item matrix, and similarity is the cosine
of the angle between two columns.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_array
from sklearn.neighbors import NearestNeighbors


def create_X(df):
    """
    Builds a sparse user-item matrix from a ratings dataframe.

    Args:
        df: pandas dataframe with columns userId, movieId, rating

    Returns:
        X: sparse array of shape (n_users, n_movies)
        user_mapper: dict that maps user id to row index
        movie_mapper: dict that maps movie id to column index
        user_inv_mapper: dict that maps row index to user id
        movie_inv_mapper: dict that maps column index to movie id
    """
    user_ids = np.unique(df["userId"])
    movie_ids = np.unique(df["movieId"])

    user_mapper = {uid: i for i, uid in enumerate(user_ids)}
    movie_mapper = {mid: i for i, mid in enumerate(movie_ids)}
    user_inv_mapper = dict(enumerate(user_ids))
    movie_inv_mapper = dict(enumerate(movie_ids))

    # Vectorized lookup: searchsorted gives each id's position in the sorted unique ids
    user_index = np.searchsorted(user_ids, df["userId"].to_numpy())
    movie_index = np.searchsorted(movie_ids, df["movieId"].to_numpy())

    X = csr_array(
        (df["rating"].to_numpy(), (user_index, movie_index)),
        shape=(len(user_ids), len(movie_ids)),
    )
    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper


class ItemRecommender:
    """Recommends movies similar to a given movie with cosine k-nearest neighbors."""

    def __init__(self, X, movie_mapper, movie_inv_mapper, center=True):
        """
        Args:
            X: sparse user-item matrix from create_X
            movie_mapper, movie_inv_mapper: mappers from create_X
            center: subtract each user's mean rating first, so a harsh rater's
                3.0 and a generous rater's 4.5 count as the same signal
        """
        self.movie_mapper = movie_mapper
        self.movie_inv_mapper = movie_inv_mapper

        X = csr_array(X, dtype=np.float64)
        # Ratings per movie, counted before centering can turn a rating into 0
        self.n_ratings = np.diff(X.T.tocsr().indptr)
        if center:
            X = _center_rows(X)

        # One row per movie; transposing keeps the matrix sparse
        self.item_matrix = X.T.tocsr()

        self.model = NearestNeighbors(metric="cosine", algorithm="brute")
        self.model.fit(self.item_matrix)

    def similar(self, movie_id, k=10, min_ratings=50):
        """
        Returns up to k movies most similar to movie_id, best first.

        Args:
            movie_id: MovieLens movie id
            k: number of movies to return
            min_ratings: skip movies with fewer ratings than this; a movie
                rated by two people can look "identical" to almost anything

        Returns:
            list of (movie_id, similarity) tuples, similarity in [-1, 1]
        """
        if movie_id not in self.movie_mapper:
            raise KeyError(f"movie id {movie_id} has no ratings")

        row = self.movie_mapper[movie_id]
        # Ask for extra neighbors, since some are filtered out below
        n_neighbors = min(self.item_matrix.shape[0], (k + 1) * 5)
        distances, indices = self.model.kneighbors(
            self.item_matrix[[row]], n_neighbors=n_neighbors
        )

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == row or self.n_ratings[idx] < min_ratings:
                continue
            results.append((int(self.movie_inv_mapper[idx]), round(float(1 - dist), 4)))
            if len(results) == k:
                break
        return results


def _center_rows(X):
    """Subtracts each user's mean from the ratings they gave; unrated cells stay empty."""
    X = X.copy()
    counts = np.diff(X.indptr)
    means = np.divide(X.sum(axis=1), counts, out=np.zeros(len(counts)), where=counts > 0)
    X.data = X.data - np.repeat(means, counts)
    return X


def titles(results, movies):
    """Adds titles to similar() results using the movies dataframe."""
    title_by_id = movies.set_index("movieId")["title"]
    return pd.DataFrame(
        [(title_by_id.get(mid, "?"), sim) for mid, sim in results],
        columns=["title", "similarity"],
    )
