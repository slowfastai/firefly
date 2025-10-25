import time
import random
import re
from typing import List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from loguru import logger
from urllib.parse import urlparse, parse_qs


SP_BASE = "https://www.startpage.com"
SP_SEARCH_SP = f"{SP_BASE}/sp/search"


def preprocess_startpage_query(q: str) -> str:
    """Normalize query for Startpage.

    Keep behavior similar to DuckDuckGo preprocessing to reduce overly
    restrictive filters and redundant site: tokens. Negated filters preserved.
    """
    try:
        # Remove wildcard prefix after site:
        text = re.sub(r"(?i)(\bsite:\s*)\*\.", r"\1", q)

        # Drop positive filetype/html filters; keep negatives
        def _drop_positive_filetype_html(m: re.Match) -> str:
            sign = m.group("sign") or ""
            return m.group(0) if sign == "-" else ""

        text = re.sub(
            r"(?i)(?P<sign>-?)\b(?:filetype|ext)\s*:\s*(?:html|htm|pdf)\b",
            _drop_positive_filetype_html,
            text,
        )

        # Clean stray boolean ops
        text = re.sub(r"(?i)^\s*(?:OR|AND)\s+", "", text)
        text = re.sub(r"(?i)\s+(?:OR|AND)\s*$", "", text)
        text = re.sub(r"(?i)\b(?:OR|AND)\s+(?:OR|AND)\b", " OR ", text)

        site_pat = re.compile(r"(?i)(?<!\w)(?P<neg>-?)site:\s*(?P<value>[^\s]+)")

        def normalize_site_value(raw: str) -> str:
            v = raw.strip()
            while (len(v) >= 2) and (
                (v[0] == v[-1] and v[0] in ['"', "'"]) or (v[0] == "(" and v[-1] == ")")
            ):
                v = v[1:-1].strip()
            v = re.sub(r"(?i)^(?:https?|ftp)://", "", v)
            for sep in ["/", "?", "#"]:
                if sep in v:
                    v = v.split(sep, 1)[0]
            if v.lower().startswith("www."):
                v = v[4:]
            v = v.rstrip(" .;,)")
            v = v.lstrip(".")
            return v.lower()

        matches = list(site_pat.finditer(text))
        if not matches:
            return text

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
                }
            )
            if (not neg) and domain:
                positive_domains.add(domain)

        keep_positive = set()
        for d in positive_domains:
            is_sub = any((e != d) and d.endswith("." + e) for e in positive_domains)
            if not is_sub:
                keep_positive.add(d)

        to_remove = set()

        def keep_score(raw: str, domain: str) -> tuple:
            rv = raw.strip()
            while (len(rv) >= 2) and (
                (rv[0] == rv[-1] and rv[0] in ['"', "'"]) or (rv[0] == "(" and rv[-1] == ")")
            ):
                rv = rv[1:-1].strip()
            rv_lower = rv.lower()
            exact = 1 if rv_lower == domain else 0
            has_scheme_or_path = 1 if re.search(r"://|/|\?|#", rv_lower) else 0
            clean_host_only = 1 if (not has_scheme_or_path) else 0
            return (exact, clean_host_only)

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
                to_remove.update(idxs)
                continue
            best = None
            best_score = None
            for i in idxs:
                raw = infos[i]["raw"]
                score = keep_score(raw, dom)
                if (best is None) or (score > best_score):
                    best = i
                    best_score = score
            for i in idxs:
                if i != best:
                    to_remove.add(i)

        if not to_remove:
            return text

        pieces = []
        last = 0
        for i, info in enumerate(infos):
            if i in to_remove:
                pieces.append(text[last : info["start"]])
                last = info["end"]
        pieces.append(text[last:])
        new_text = "".join(pieces)
        new_text = re.sub(r"\s{2,}", " ", new_text).strip()
        return new_text or text
    except Exception:
        return q


def _extract_startpage_link(href: str) -> Optional[str]:
    """Extract a usable external URL from a Startpage result href.

    - Accept absolute http/https links directly.
    - For Startpage redirect pattern like `/rd?l=<encoded>`, extract `l`.
    - Skip internal or fragment-only links.
    """
    if not href:
        return None
    if href.startswith("#"):
        return None
    if href.startswith("http://") or href.startswith("https://"):
        # External URL already
        return href

    # Try to decode Startpage redirect links
    try:
        if href.startswith("/rd") or href.startswith("/do/rd"):
            parsed = urlparse(href)
            q = parse_qs(parsed.query)
            link = q.get("l") or q.get("u") or q.get("q")
            if link and isinstance(link, list) and link[0].startswith(("http://", "https://")):
                return link[0]
    except Exception:
        pass

    # Discard other relative/internal links
    return None


def search(
    query: str,
    user_agent: Optional[str] = None,
    max_results: int = 10,
    timeout: int = 20,
) -> List[str]:
    """
    Perform a Startpage search and return a list of result URLs.

    Uses the public HTML endpoints; no API key required.
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
        "Referer": SP_BASE,
    }

    processed = preprocess_startpage_query(query)
    params = {
        # Keep minimal params for lower bot-signal surface
        "query": processed,
    }
    logger.info(f"Startpage search query: {processed}")

    # Preflight to seed cookies/session
    try:
        session.get(SP_BASE, headers=headers, timeout=timeout)
    except Exception as e:
        logger.debug(f"Startpage preflight GET failed (continuing): {e}")

    def _fetch_and_parse(url: str, q_params: dict) -> Tuple[List[str], bool]:
        try:
            r = session.get(url, headers=headers, params=q_params, timeout=timeout)
            r.raise_for_status()
            html = r.text
            lowered = html.lower()
            if (
                ("captcha" in lowered)
                or ("unusual traffic" in lowered)
                or ("verify you are a human" in lowered)
                or ("enable javascript" in lowered)
            ):
                logger.warning(
                    "Startpage returned a page indicating possible rate limiting or bot check"
                )
                return [], True

            soup = BeautifulSoup(html, "html.parser")
            out: List[str] = []
            seen = set()
            # Prefer explicit result anchors; fall back to broad selection
            for a in soup.select(
                "a[data-testid=\"result-title-a\"], a.w-gl__result-title, a.result-link, a[href]"
            ):
                href = a.get("href")
                url2 = _extract_startpage_link(href)
                if not url2:
                    continue
                # Filter internal links
                try:
                    netloc = urlparse(url2).netloc
                    if not netloc or "startpage.com" in netloc:
                        continue
                except Exception:
                    continue
                if url2 in seen:
                    continue
                seen.add(url2)
                out.append(url2)
                if len(out) >= max_results:
                    break
            return out, False
        except Exception as e:
            logger.debug(f"Startpage fetch error: {e}")
            return [], True

    # Try the newer `/sp/search` first
    logger.info("Startpage: trying /sp/search via GET")
    results, had_issue = _fetch_and_parse(SP_SEARCH_SP, params)

    # Soft relax: if still empty and quotes present, remove quotes once
    if not results and (('"' in processed) or ("'" in processed)):
        relaxed = processed.replace('"', "").replace("'", "")
        if relaxed != processed:
            logger.info(f"Startpage fallback: removing quotes -> {relaxed}")
            params_relaxed = dict(params)
            params_relaxed["query"] = relaxed
            logger.info("Startpage: relaxed retry on /sp/search (quotes removed)")
            results_relaxed, had_issue_relaxed = _fetch_and_parse(SP_SEARCH_SP, params_relaxed)
            if results_relaxed:
                results = results_relaxed
            had_issue = had_issue or had_issue_relaxed

    # If still empty and we saw a bot-check/CAPTCHA page, raise to trigger upstream fallback
    if not results and had_issue:
        raise RuntimeError("Startpage returned CAPTCHA/bot-check page; no results")

    # Gentle pause
    time.sleep(random.uniform(0.6, 1.2))
    logger.info(f"Startpage results: {results}")
    return results


if __name__ == "__main__":
    search_query = (
        'site:mozilla.org OR site:w3.org "privacy policy" -filetype:html -ext:htm'
    )
    print(search_query)
    print(preprocess_startpage_query(search_query))
