import numpy as np
import pandas as pd
import pytest

from recommender import ItemRecommender, create_X


def make_ratings():
    # Users 1-3 love movies 10 and 20 and dislike 30; users 4-5 the opposite.
    rows = []
    for user in (1, 2, 3):
        rows += [(user, 10, 5.0), (user, 20, 4.5), (user, 30, 1.0)]
    for user in (4, 5):
        rows += [(user, 10, 1.0), (user, 20, 1.5), (user, 30, 5.0)]
    rows.append((6, 40, 3.0))  # movie 40 has a single rating
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating"])


def test_create_X_shape_and_mappers():
    X, user_mapper, movie_mapper, user_inv, movie_inv = create_X(make_ratings())
    assert X.shape == (6, 4)
    assert X.nnz == 16
    assert X[user_mapper[1], movie_mapper[10]] == 5.0
    assert all(movie_inv[movie_mapper[m]] == m for m in (10, 20, 30, 40))
    assert all(user_inv[user_mapper[u]] == u for u in range(1, 7))


def test_create_X_works_with_non_contiguous_ids():
    df = pd.DataFrame({"userId": [100, 7], "movieId": [999, 3], "rating": [4.0, 2.0]})
    X, user_mapper, movie_mapper, _, _ = create_X(df)
    assert X[user_mapper[100], movie_mapper[999]] == 4.0
    assert X[user_mapper[7], movie_mapper[3]] == 2.0


def test_similar_movie_ranks_first():
    X, _, movie_mapper, _, movie_inv = create_X(make_ratings())
    rec = ItemRecommender(X, movie_mapper, movie_inv)
    results = rec.similar(10, k=3, min_ratings=1)
    assert results[0][0] == 20
    assert results[0][1] > 0.9
    # Movie 30 is liked by the opposite group, so it is least similar
    assert results[-1][0] == 30
    assert results[-1][1] < 0


def test_never_returns_the_query_movie():
    X, _, movie_mapper, _, movie_inv = create_X(make_ratings())
    rec = ItemRecommender(X, movie_mapper, movie_inv)
    assert 10 not in [mid for mid, _ in rec.similar(10, k=3, min_ratings=1)]


def test_min_ratings_filters_rare_movies():
    X, _, movie_mapper, _, movie_inv = create_X(make_ratings())
    rec = ItemRecommender(X, movie_mapper, movie_inv)
    assert 40 not in [mid for mid, _ in rec.similar(10, k=3, min_ratings=2)]


def test_unknown_movie_raises():
    X, _, movie_mapper, _, movie_inv = create_X(make_ratings())
    rec = ItemRecommender(X, movie_mapper, movie_inv)
    with pytest.raises(KeyError):
        rec.similar(12345)
