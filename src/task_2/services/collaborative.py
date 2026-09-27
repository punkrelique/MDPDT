import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix

TYPE_WEIGHTS = {"clicks": 1, "carts": 3, "orders": 5}


def build_interaction_matrix(events: pd.DataFrame):
    sessions = events["session"].astype("category")
    items = events["aid"].astype("category")
    weights = events["type"].map(TYPE_WEIGHTS).fillna(1).astype(float)

    matrix = coo_matrix(
        (weights, (sessions.cat.codes, items.cat.codes)),
        shape=(len(sessions.cat.categories), len(items.cat.categories)),
    ).tocsr()
    return matrix, sessions.cat.categories, items.cat.categories


def train_als(
    matrix, factors: int = 64, regularization: float = 0.01, iterations: int = 15
) -> AlternatingLeastSquares:
    model = AlternatingLeastSquares(
        factors=factors, regularization=regularization, iterations=iterations
    )
    model.fit(matrix)
    return model


def recommend_for_session(
    model: AlternatingLeastSquares, matrix, session_idx: int, item_categories, top_n: int = 20
) -> list[int]:
    ids, _scores = model.recommend(session_idx, matrix[session_idx], N=top_n)
    return [int(item_categories[i]) for i in ids]
