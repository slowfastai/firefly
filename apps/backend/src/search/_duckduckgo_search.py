import time
import random
from typing import List, Optional

import requests
import re
from bs4 import BeautifulSoup
from loguru import logger


DDG_HTML = "https://html.duckduckgo.com/html/"


def preprocess_duckduckgo_query(q: str) -> str:
    """Normalize query for DuckDuckGo limitations.

    - Convert patterns like `site:*.gov` -> `site:gov` (and similarly for other TLDs).
    - Remove redundant positive site filters: if both `site:root.com` and
      `site:sub.root.com` appear, keep only `site:root.com`.
      Negated filters (e.g., `-site:`) are preserved as-is to avoid changing
      exclusion semantics.
    """
    try:
        # 1) Remove wildcard prefix after `site:` (case-insensitive, allow optional whitespace)
        text = re.sub(r"(?i)(\bsite:\s*)\*\.", r"\1", q)

        # 2) Find all site: filters and drop redundant positives where a parent domain exists
        site_pat = re.compile(r"(?i)(?<!\w)(?P<neg>-?)site:\s*(?P<value>[^\s]+)")

        def normalize_site_value(raw: str) -> str:
            v = raw.strip()
            # strip surrounding quotes or parentheses if they wrap the whole token
            while (len(v) >= 2) and ((v[0] == v[-1] and v[0] in ['"', "'"]) or (v[0] == '(' and v[-1] == ')')):
                v = v[1:-1].strip()
            # strip URL scheme
            v = re.sub(r"(?i)^(?:https?|ftp)://", "", v)
            # take host part only (before /, ?, #)
            for sep in ['/', '?', '#']:
                if sep in v:
                    v = v.split(sep, 1)[0]
            # remove leading www.
            if v.lower().startswith('www.'):
                v = v[4:]
            # remove trailing dot(s) and stray punctuation
            v = v.rstrip(' .;,)')
            v = v.lstrip('.')
            return v.lower()

        matches = list(site_pat.finditer(text))
        if not matches:
            return text

        # Collect normalized domains
        infos = []
        positive_domains = set()
        for m in matches:
            neg = bool(m.group('neg'))
            raw_val = m.group('value')
            domain = normalize_site_value(raw_val)
            infos.append({
                'start': m.start(),
                'end': m.end(),
                'neg': neg,
                'domain': domain,
                'raw': raw_val,
                'index': len(infos),
            })
            if (not neg) and domain:
                positive_domains.add(domain)

        # Keep only broadest positive domains (drop subdomains if parent present)
        keep_positive = set()
        for d in positive_domains:
            is_sub = any((e != d) and d.endswith('.' + e) for e in positive_domains)
            if not is_sub:
                keep_positive.add(d)

        # Build removal plan with preference for canonical tokens per kept domain
        to_remove = set()

        # Helper to score which token to keep for a given domain
        def keep_score(raw: str, domain: str) -> tuple:
            # Strip outer quotes/parentheses for comparison
            rv = raw.strip()
            while (len(rv) >= 2) and ((rv[0] == rv[-1] and rv[0] in ['"', "'"]) or (rv[0] == '(' and rv[-1] == ')')):
                rv = rv[1:-1].strip()
            rv_lower = rv.lower()
            # Exact domain text (best)
            exact = 1 if rv_lower == domain else 0
            # Host-only without scheme/path/query/fragment
            has_scheme_or_path = 1 if re.search(r"://|/|\?|#", rv_lower) else 0
            clean_host_only = 1 if (not has_scheme_or_path) else 0
            # Prefer exact match > clean host-only > others
            return (exact, clean_host_only)

        # Map domain -> list of positive token indices for that domain
        domain_to_indices = {}
        for i, info in enumerate(infos):
            if info['neg']:
                continue
            dom = info['domain']
            if not dom:
                continue
            domain_to_indices.setdefault(dom, []).append(i)

        for dom, idxs in domain_to_indices.items():
            if dom not in keep_positive:
                # remove all occurrences of subdomains that are not kept
                to_remove.update(idxs)
                continue
            # Choose best token to keep for this domain
            best = None
            best_score = None
            for i in idxs:
                raw = infos[i]['raw']
                score = keep_score(raw, dom)
                if (best is None) or (score > best_score) or (score == best_score and infos[i]['start'] < infos[best]['start']):
                    best = i
                    best_score = score
            # Remove all others except the best
            for i in idxs:
                if i != best:
                    to_remove.add(i)

        if not to_remove:
            return text

        # Reconstruct text without removed site tokens
        pieces = []
        last = 0
        for i, info in enumerate(infos):
            if i in to_remove:
                pieces.append(text[last:info['start']])
                last = info['end']
        pieces.append(text[last:])
        new_text = ''.join(pieces)

        # Light whitespace cleanup
        new_text = re.sub(r"\s{2,}", " ", new_text).strip()
        return new_text or text
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
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": DDG_HTML,
    }
    processed = preprocess_duckduckgo_query(query)
    params = {
        "q": processed,
        "kp": "-1",  # Moderate Safe Search, "1" is "Strict Safe Search"
        "kl": "us-en",
        "ia": "web",  # Only web results
        # "t": "slowfast-firefly",  # identifier
        }
    logger.info(f"DuckDuckGo search query: {processed}")
    # Preflight GET to obtain minimal cookies/session; helps avoid empty/blocked responses
    try:
        session.get(DDG_HTML, headers=headers, timeout=timeout)
    except Exception as e:
        logger.debug(f"DDG preflight GET failed (continuing): {e}")

    r = session.post(DDG_HTML, headers=headers, data=params, timeout=timeout)
    r.raise_for_status()
    html = r.text
    lowered = html.lower()
    if (
        ("captcha" in lowered)
        or ("unusual traffic" in lowered)
        or ("verify you are a human" in lowered)
    ):
        logger.warning("DuckDuckGo returned a page indicating possible rate limiting or bot check")
        return []

    soup = BeautifulSoup(html, "html.parser")
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

if __name__ == "__main__":
    search_query = 'site:google.com/advanced_search OR site:searchenginewatch.com OR site:moz.com "search operators" "site:" "OR" "filetype:"'
    print(search_query)
    print(preprocess_duckduckgo_query(search_query))

    search_query = 'DuckDuckGo 2024 2025 future features AI duck.ai roadmap site:duckduckgo.com OR site:blog.duckduckgo.com'
    print(search_query)
    print(preprocess_duckduckgo_query(search_query))
