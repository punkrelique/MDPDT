import pandas as pd

TYPE_WEIGHTS = {"clicks": 1, "carts": 3, "orders": 5}


def popularity(events: pd.DataFrame, top_n: int = 20) -> pd.Series:
    weights = events["type"].map(TYPE_WEIGHTS).fillna(1)
    scores = weights.groupby(events["aid"]).sum()
    return scores.sort_values(ascending=False).head(top_n)


def covisitation_counts(
    events: pd.DataFrame, top_n: int = 20, max_events_per_session: int = 30
) -> dict[int, list[int]]:
    trimmed = events.sort_values("ts").groupby("session").tail(max_events_per_session)
    merged = trimmed.merge(trimmed, on="session")
    merged = merged[merged["aid_x"] != merged["aid_y"]]
    counts = merged.groupby(["aid_x", "aid_y"]).size()

    result: dict[int, list[int]] = {}
    for aid_x, group in counts.groupby(level=0):
        top = group.sort_values(ascending=False).head(top_n)
        result[aid_x] = [aid_y for _, aid_y in top.index]
    return result
