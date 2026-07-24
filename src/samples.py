"""The curated sample corpus — "A Deep Dive into LLMs" (the read-only / page).

Four visually-rich LLM talks. The worker auto-ingests any that aren't indexed
when it boots (SEED_SAMPLE_VIDEOS=true, the default), so a fresh clone is
queryable the moment the stack comes up; examples/quickstart.py uses the same
list for the in-process route.
"""
from __future__ import annotations

import re

SAMPLE_VIDEOS = [
    {
        "url": "https://youtu.be/LPZh9BOjkQs",
        "title": "3Blue1Brown — LLMs explained briefly (8m)",
    },
    {
        "url": "https://youtu.be/wjZofJX0v4M",
        "title": "3Blue1Brown — Transformers, the tech behind LLMs (27m)",
    },
    {
        "url": "https://youtu.be/eMlx5fFNoYc",
        "title": "3Blue1Brown — Attention in transformers, step-by-step (26m)",
    },
    {
        "url": "https://youtu.be/zjkBMFhNj_g",
        "title": "Andrej Karpathy — [1hr Talk] Intro to Large Language Models (60m)",
    },
]


def sample_video_id(url: str) -> str:
    m = re.search(r"(?:youtu\.be/|v=)([\w-]{11})", url)
    return f"yt_{m.group(1)}" if m else url


# The four sample ids — protected: they can be unselected from a query but
# never deleted (the seed gate would just re-add them anyway).
SAMPLE_IDS = frozenset(sample_video_id(v["url"]) for v in SAMPLE_VIDEOS)


def is_sample(video_id: str) -> bool:
    return video_id in SAMPLE_IDS
