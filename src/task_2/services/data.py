import json
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd


def iter_sessions(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_events(path: Path, max_sessions: Optional[int] = None) -> pd.DataFrame:
    rows = []
    for i, session in enumerate(iter_sessions(path)):
        if max_sessions is not None and i >= max_sessions:
            break
        session_id = session["session"]
        for event in session["events"]:
            rows.append((session_id, event["aid"], event["ts"], event["type"]))
    return pd.DataFrame(rows, columns=["session", "aid", "ts", "type"])
