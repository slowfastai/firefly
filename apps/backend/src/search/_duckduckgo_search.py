import time
import random
from typing import List, Optional, Tuple

import requests
import re
from bs4 import BeautifulSoup
from loguru import logger


DDG_HTML = "https://html.duckduckgo.com/html/"
DDG_LITE = "https://lite.duckduckgo.com/lite/"


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

        # 1.1) Drop positive filetype/html filters (usually overly restrictive)
        # Keep negative forms (e.g., -filetype:html) to preserve user intent.
        def _drop_positive_filetype_html(m: re.Match) -> str:
            sign = m.group("sign") or ""
            return m.group(0) if sign == "-" else ""

        text = re.sub(
            r"(?i)(?P<sign>-?)\b(?:filetype|ext)\s*:\s*(?:html|htm)\b",
            _drop_positive_filetype_html,
            text,
        )

        # Clean up leading/trailing stray boolean operators after removals
        text = re.sub(r"(?i)^\s*(?:OR|AND)\s+", "", text)
        text = re.sub(r"(?i)\s+(?:OR|AND)\s*$", "", text)
        text = re.sub(r"(?i)\b(?:OR|AND)\s+(?:OR|AND)\b", " OR ", text)

        # 2) Find all site: filters and drop redundant positives where a parent domain exists
        site_pat = re.compile(r"(?i)(?<!\w)(?P<neg>-?)site:\s*(?P<value>[^\s]+)")

        def normalize_site_value(raw: str) -> str:
            v = raw.strip()
            # strip surrounding quotes or parentheses if they wrap the whole token
            while (len(v) >= 2) and (
                (v[0] == v[-1] and v[0] in ['"', "'"]) or (v[0] == "(" and v[-1] == ")")
            ):
                v = v[1:-1].strip()
            # strip URL scheme
            v = re.sub(r"(?i)^(?:https?|ftp)://", "", v)
            # take host part only (before /, ?, #)
            for sep in ["/", "?", "#"]:
                if sep in v:
                    v = v.split(sep, 1)[0]
            # remove leading www.
            if v.lower().startswith("www."):
                v = v[4:]
            # remove trailing dot(s) and stray punctuation
            v = v.rstrip(" .;,)")
            v = v.lstrip(".")
            return v.lower()

        matches = list(site_pat.finditer(text))
        if not matches:
            return text

        # Collect normalized domains
        infos = []
        positive_domains = set()
        for m in matches:
            neg = bool(m.group("neg"))
            raw_val = m.group("value")
            domain = normalize_site_value(raw_val)
            infos.append(
                {
                    "start": m.start(),
                    "end": m.end(),
                    "neg": neg,
                    "domain": domain,
                    "raw": raw_val,
                    "index": len(infos),
                }
            )
            if (not neg) and domain:
                positive_domains.add(domain)

        # Keep only broadest positive domains (drop subdomains if parent present)
        keep_positive = set()
        for d in positive_domains:
            is_sub = any((e != d) and d.endswith("." + e) for e in positive_domains)
            if not is_sub:
                keep_positive.add(d)

        # Build removal plan with preference for canonical tokens per kept domain
        to_remove = set()

        # Helper to score which token to keep for a given domain
        def keep_score(raw: str, domain: str) -> tuple:
            # Strip outer quotes/parentheses for comparison
            rv = raw.strip()
            while (len(rv) >= 2) and (
                (rv[0] == rv[-1] and rv[0] in ['"', "'"])
                or (rv[0] == "(" and rv[-1] == ")")
            ):
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
            if info["neg"]:
                continue
            dom = info["domain"]
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
                raw = infos[i]["raw"]
                score = keep_score(raw, dom)
                if (
                    (best is None)
                    or (score > best_score)
                    or (
                        score == best_score and infos[i]["start"] < infos[best]["start"]
                    )
                ):
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
                pieces.append(text[last : info["start"]])
                last = info["end"]
        pieces.append(text[last:])
        new_text = "".join(pieces)

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

    def _fetch_and_parse(data_params) -> Tuple[List[str], bool]:
        """Return (results, had_issue) where had_issue indicates HTTP errors or bot-check page."""
        try:
            r_ = session.post(
                DDG_HTML, headers=headers, data=data_params, timeout=timeout
            )
            r_.raise_for_status()
            html_ = r_.text
            lowered_ = html_.lower()
            if (
                ("captcha" in lowered_)
                or ("unusual traffic" in lowered_)
                or ("verify you are a human" in lowered_)
            ):
                logger.warning(
                    "DuckDuckGo returned a page indicating possible rate limiting or bot check"
                )
                return [], True

            soup_ = BeautifulSoup(html_, "html.parser")
            out: List[str] = []
            for a in soup_.select("a.result__a"):
                url = a.get("href")
                title = a.get_text(" ", strip=True)
                if url and title:
                    out.append(url)
                if len(out) >= max_results:
                    break
            return out, False
        except Exception as e:
            logger.debug(f"DDG HTML fetch error: {e}")
            return [], True

    results, had_issue = _fetch_and_parse(params)

    # Fallback: if no results and there are quotes, remove all quotes once
    if not results and (('"' in processed) or ("'" in processed)):
        relaxed = processed.replace('"', "").replace("'", "")
        if relaxed != processed:
            logger.info(f"DDG fallback: removing quotes -> {relaxed}")
            params_relaxed = dict(params)
            params_relaxed["q"] = relaxed
            results, had_issue = _fetch_and_parse(params_relaxed)
    # If still empty due to HTML issues, try DDG Lite (lighter HTML endpoint via GET)
    if not results and had_issue:
        try:
            logger.info("DDG fallback: trying DuckDuckGo Lite endpoint")
            # Use a conservative GET with the same headers
            lite_params = {"q": processed}
            # Preflight GET to seed cookies if helpful
            try:
                session.get(DDG_LITE, headers=headers, timeout=timeout)
            except Exception:
                pass

            r_lite = session.get(
                DDG_LITE, headers=headers, params=lite_params, timeout=timeout
            )
            r_lite.raise_for_status()
            html_lite = r_lite.text
            lowered_lite = html_lite.lower()
            if (
                ("captcha" in lowered_lite)
                or ("unusual traffic" in lowered_lite)
                or ("verify you are a human" in lowered_lite)
            ):
                logger.warning(
                    "DDG Lite returned a page indicating possible rate limiting or bot check"
                )
            else:
                soup_lite = BeautifulSoup(html_lite, "html.parser")
                # Primary: common lite selectors may vary; prefer explicit result anchors if present
                out_lite: List[str] = []
                for a in soup_lite.select("a.result-link, a[href]"):
                    href = a.get("href")
                    title = a.get_text(" ", strip=True)
                    if not href or not title:
                        continue
                    # Prefer absolute links; skip internal anchors
                    if href.startswith("http://") or href.startswith("https://"):
                        out_lite.append(href)
                    if len(out_lite) >= max_results:
                        break
                results = out_lite
        except Exception as e:
            logger.debug(f"DDG Lite fallback failed: {e}")

    # Gentle pause to avoid rate issues
    time.sleep(random.uniform(0.6, 1.2))
    logger.info(f"DuckDuckGo results: {results}")
    return results


if __name__ == "__main__":
    search_query = 'site:google.com/advanced_search OR site:searchenginewatch.com OR site:moz.com "search operators" "site:" "OR" "filetype:"'
    print(search_query)
    print(preprocess_duckduckgo_query(search_query))

    search_query = "DuckDuckGo 2024 2025 future features AI duck.ai roadmap site:duckduckgo.com OR site:blog.duckduckgo.com"
    print(search_query)
    print(preprocess_duckduckgo_query(search_query))
