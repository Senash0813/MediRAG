from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


S2_FIELDS = ",".join(
    [
        "title",
        "year",
        "venue",
        "citationCount",
        "influentialCitationCount",
        "publicationTypes",
    ]
)


def extract_corpus_id(paper_url: Optional[str]):
    if not paper_url or "CorpusID:" not in paper_url:
        return None
    return paper_url.split("CorpusID:")[-1]


def fallback_s2_metadata(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": doc.get("paper_title"),
        "year": doc.get("year"),
        "venue": doc.get("venue"),
        "citationCount": 0,
        "influentialCitationCount": 0,
        "publicationTypes": [],
        "_fallback": True,
    }


@dataclass
class SemanticScholarClient:
    api_key: str

    def __post_init__(self):
        self._headers = {"x-api-key": self.api_key}

    def fetch_metadata(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        corpus_id = extract_corpus_id(doc.get("paper_url"))
        if not corpus_id:
            return fallback_s2_metadata(doc)

        url = f"https://api.semanticscholar.org/graph/v1/paper/CorpusID:{corpus_id}"
        params = {"fields": S2_FIELDS}

        r = requests.get(url, headers=self._headers, params=params)
        time.sleep(1.1)  # rate limit 1 request/sec

        if r.status_code != 200:
            return fallback_s2_metadata(doc)

        return r.json()
