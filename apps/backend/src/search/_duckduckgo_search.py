import time
import random
import urllib.parse
from typing import List, Optional

import requests
import re
from bs4 import BeautifulSoup
from loguru import logger


DDG_HTML = "https://html.duckduckgo.com/html/"


def preprocess_duckduckgo_query(q: str) -> str:
    """Normalize query for DuckDuckGo limitations.

    - Convert patterns like `site:*.gov` -> `site:gov` (and similarly for other TLDs).
    - Works generically by removing the leading `*.` after `site:`.
    """
    try:
        # Remove wildcard prefix after `site:` (case-insensitive, allow optional whitespace)
        q2 = re.sub(r"(?i)(\bsite:\s*)\*\.", r"\1", q)
        return q2
    except Exception:
        return q


def search(
    query: str,
    user_agent: Optional[str] = None,
    max_results: int = 10,
    timeout: int = 20,
) -> List[str]:
    """
    Perform a DuckDuckGo HTML search and return a list of result URLs.

    Uses lightweight HTML endpoint; no API key required.
    """
    session = requests.Session()
    headers = {
        "User-Agent": user_agent
        or (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }
    processed = preprocess_duckduckgo_query(query)
    data = {"q": processed}
    logger.info(f"DuckDuckGo search query: {processed}")
    r = session.post(DDG_HTML, headers=headers, data=data, timeout=timeout)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    results: List[str] = []
    for a in soup.select("a.result__a"):
        url = a.get("href")
        title = a.get_text(" ", strip=True)
        if url and title:
            # results.append({"title": title, "url": url})  # TODO: add title to results?
            results.append(url)
        if len(results) >= max_results:
            break
    # Gentle pause to avoid rate issues
    time.sleep(random.uniform(0.6, 1.2))
    logger.info(f"DuckDuckGo results: {results}")
    return results
