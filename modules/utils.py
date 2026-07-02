# app/utils/logger.py

import json
import os
from datetime import datetime, timezone

_LOG_PATH = "data/query_log.jsonl"


def log_query(
    doc_names: list[str],
    query: str,
    top_k: int,
    grounded: bool,
    num_results: int,
) -> None:
    os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "documents": doc_names,
        "query": query,
        "top_k": top_k,
        "grounded": grounded,
        "num_results": num_results,
    }
    with open(_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def read_recent_logs(limit: int = 20) -> list[dict]:
    if not os.path.exists(_LOG_PATH):
        return []
    with open(_LOG_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()[-limit:]
    return [json.loads(line) for line in lines]
