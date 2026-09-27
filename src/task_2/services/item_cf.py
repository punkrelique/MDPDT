import numpy as np
from scipy.sparse import csc_matrix


def build_item_similarity(matrix) -> tuple[csc_matrix, np.ndarray]:
    item_matrix = matrix.tocsc()
    squared_norms = np.asarray(item_matrix.multiply(item_matrix).sum(axis=0)).ravel()
    norms = np.sqrt(squared_norms)
    norms[norms == 0] = 1.0
    return item_matrix, norms


def top_similar_items(item_matrix: csc_matrix, norms: np.ndarray, item_idx: int, top_n: int = 20) -> list[int]:
    item_vector = item_matrix[:, item_idx]
    scores = np.asarray((item_matrix.T @ item_vector).todense()).ravel()
    scores = scores / (norms * norms[item_idx])
    scores[item_idx] = -np.inf

    if top_n >= len(scores):
        top_idx = np.argsort(-scores)
    else:
        top_idx = np.argpartition(scores, -top_n)[-top_n:]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
    return top_idx.tolist()
