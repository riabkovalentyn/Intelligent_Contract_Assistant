from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger("feedback")

_FEEDBACK_FILE = os.path.join(config.data_dir, "feedback.jsonl")


def record_feedback(
    session_id: str,
    question: str,
    answer: str,
    rating: int,
    notes: Optional[str],
    sources: Optional[List[Dict[str, Any]]] = None,
) -> None:
    os.makedirs(os.path.dirname(_FEEDBACK_FILE), exist_ok=True)
    row = {
        "ts": int(time.time()),
        "session": session_id,
        "question": question,
        "answer": answer,
        "rating": rating,
        "notes": notes,
        "sources": sources or [],
    }
    with open(_FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Feedback appended to %s", _FEEDBACK_FILE)