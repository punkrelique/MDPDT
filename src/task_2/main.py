import argparse
from pathlib import Path

from services import baselines, collaborative, data, evaluate, item_cf

DATA_DIR = Path(__file__).parent / "data"


def load(data_file: str, max_sessions: int | None):
    return data.load_events(DATA_DIR / data_file, max_sessions=max_sessions)


def cmd_evaluate(args):
    events = load(args.data_file, args.max_sessions)
    train, test = evaluate.train_test_split_sessions(events)

    pop = baselines.popularity(train, top_n=args.top_n)
    pop_ranking = pop.index.tolist()
    pop_preds = {session: pop_ranking for session in test["session"].unique()}
    print(f"popularity recall@{args.top_n}: {evaluate.recall_at_k(pop_preds, test, k=args.top_n):.4f}")

    covis = baselines.covisitation_counts(train, top_n=args.top_n)
    last_item_per_session = train.sort_values("ts").groupby("session")["aid"].last()
    covis_preds = {
        session: covis.get(aid, pop_ranking) for session, aid in last_item_per_session.items()
    }
    print(f"covisitation recall@{args.top_n}: {evaluate.recall_at_k(covis_preds, test, k=args.top_n):.4f}")

    matrix, session_categories, item_categories = collaborative.build_interaction_matrix(train)
    item_index = {item: idx for idx, item in enumerate(item_categories)}

    model = collaborative.train_als(matrix)
    session_index = {session: idx for idx, session in enumerate(session_categories)}
    als_preds = {}
    for session in test["session"].unique():
        idx = session_index.get(session)
        als_preds[session] = (
            pop_ranking
            if idx is None
            else collaborative.recommend_for_session(model, matrix, idx, item_categories, top_n=args.top_n)
        )
    print(f"als recall@{args.top_n}: {evaluate.recall_at_k(als_preds, test, k=args.top_n):.4f}")

    item_matrix, norms = item_cf.build_item_similarity(matrix)
    itemcf_preds = {}
    for session, aid in last_item_per_session.items():
        idx = item_index.get(aid)
        itemcf_preds[session] = (
            pop_ranking
            if idx is None
            else [int(item_categories[i]) for i in item_cf.top_similar_items(item_matrix, norms, idx, top_n=args.top_n)]
        )
    print(f"item-cf recall@{args.top_n}: {evaluate.recall_at_k(itemcf_preds, test, k=args.top_n):.4f}")


def cmd_recommend(args):
    events = load(args.data_file, args.max_sessions)
    pop_ranking = baselines.popularity(events, top_n=args.top_n).index.tolist()

    if args.item_id is None:
        print(f"Popular items: {pop_ranking}")
        return

    if args.method == "item-cf":
        matrix, _session_categories, item_categories = collaborative.build_interaction_matrix(events)
        item_index = {item: idx for idx, item in enumerate(item_categories)}
        idx = item_index.get(args.item_id)
        if idx is None:
            recs = pop_ranking
        else:
            item_matrix, norms = item_cf.build_item_similarity(matrix)
            similar_idx = item_cf.top_similar_items(item_matrix, norms, idx, top_n=args.top_n)
            recs = [int(item_categories[i]) for i in similar_idx]
        print(f"Recommendations for item {args.item_id}: {recs}")
        return

    covis = baselines.covisitation_counts(events, top_n=args.top_n)
    print(f"Recommendations for item {args.item_id}: {covis.get(args.item_id, pop_ranking)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OTTO RecSys: popularity/covisitation baselines + ALS/item-CF")
    parser.add_argument("--data-file", default="otto-recsys-train.jsonl", help="dataset file under ./data")
    parser.add_argument("--max-sessions", type=int, default=None, help="cap sessions loaded, for quick runs")
    parser.add_argument("--top-n", type=int, default=20)

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_evaluate = subparsers.add_parser("evaluate", help="compute recall@k for each approach")
    p_evaluate.set_defaults(func=cmd_evaluate)

    p_recommend = subparsers.add_parser("recommend", help="print recommendations")
    p_recommend.add_argument("--item-id", type=int, default=None, help="show similar/co-visited items for this aid")
    p_recommend.add_argument("--method", choices=["covisitation", "item-cf"], default="covisitation")
    p_recommend.set_defaults(func=cmd_recommend)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
