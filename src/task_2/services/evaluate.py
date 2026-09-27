import pandas as pd


def train_test_split_sessions(events: pd.DataFrame, holdout_events: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = events.sort_values(["session", "ts"])
    rank_from_end = events.groupby("session").cumcount(ascending=False)
    test_mask = rank_from_end < holdout_events
    return events[~test_mask], events[test_mask]


def recall_at_k(predictions: dict[int, list[int]], ground_truth: pd.DataFrame, k: int = 20) -> float:
    truth_by_session = ground_truth.groupby("session")["aid"].apply(set)
    scores = []
    for session, truth in truth_by_session.items():
        if not truth:
            continue
        preds = set(predictions.get(session, [])[:k])
        scores.append(len(preds & truth) / min(k, len(truth)))
    return sum(scores) / len(scores) if scores else 0.0
