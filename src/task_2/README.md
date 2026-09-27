# Task 2. RecSys

## Dataset

[OTTO Recommender Systems Dataset](https://www.kaggle.com/datasets/otto/recsys-dataset) (Kaggle handle `otto/recsys-dataset`).

E-commerce session logs. Each session is a sequence of events:

- `session` — session id
- `aid` — item id (article id)
- `ts` — event timestamp
- `type` — event type: `clicks`, `carts`, or `orders`


## Setup

```bash
cp .env.example .env
make install
make download
make run
```

## Usage

```bash
# recall@k for popularity, co-visitation, ALS, and item-based CF
.venv/bin/python main.py --max-sessions 20000 evaluate

# global popularity ranking
.venv/bin/python main.py recommend

# items most often co-visited with item 123
.venv/bin/python main.py recommend --item-id 123

# items most similar to item 123 by cosine similarity (item-based CF)
.venv/bin/python main.py recommend --item-id 123 --method item-cf
```