import time
import random
import re
from typing import List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from loguru import logger
from urllib.parse import urlparse


NAVER_BASE = "https://search.naver.com"
NAVER_SEARCH = f"{NAVER_BASE}/search.naver"


def preprocess_naver_query(q: str) -> str:
    """Normalize query for Naver.

    Keep behavior consistent with other engines:
    - Strip leading '*.' after site:
    - Drop positive filetype/html/pdf filters (keep negatives)
    - Remove redundant positive site: tokens when a parent domain exists
    """
    try:
        text = re.sub(r"(?i)(\bsite:\s*)\*\.", r"\1", q)

        # Drop positive filetype/html/pdf; keep negatives
        def _drop_positive_filetype_ext(m: re.Match) -> str:
            sign = m.group("sign") or ""
            return m.group(0) if sign == "-" else ""

        text = re.sub(
            r"(?i)(?P<sign>-?)\b(?:filetype|ext)\s*:\s*(?:html|htm|pdf)\b",
            _drop_positive_filetype_ext,
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


def _looks_like_result_link(href: str) -> Optional[str]:
    """Return a cleaned external URL if this looks like a SERP result link."""
    try:
        if not href:
            return None
        if not href.startswith(("http://", "https://")):
            return None
        parsed = urlparse(href)
        if not parsed.netloc:
            return None
        # Filter out the search result page itself; keep other naver subdomains like blog.naver.com
        if "search.naver.com" in parsed.netloc:
            return None
        return href
    except Exception:
        return None


def search(
    query: str,
    user_agent: Optional[str] = None,
    max_results: int = 10,
    timeout: int = 20,
) -> List[str]:
    """
    Perform a Naver search and return a list of result URLs.

    Uses the public HTML endpoint; no API key required.
    """
    session = requests.Session()
    headers = {
        "User-Agent": user_agent
        or (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": NAVER_BASE,
    }

    processed = preprocess_naver_query(query)
    params = {
        "query": processed,
    }
    logger.info(f"Naver search query: {processed}")

    # Preflight to seed cookies/session
    try:
        session.get(NAVER_BASE, headers=headers, timeout=timeout)
    except Exception as e:
        logger.debug(f"Naver preflight GET failed (continuing): {e}")

    def _fetch_and_parse(q_params: dict) -> Tuple[List[str], bool]:
        try:
            r = session.get(NAVER_SEARCH, headers=headers, params=q_params, timeout=timeout)
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
                    "Naver returned a page indicating possible rate limiting or bot check"
                )
                return [], True

            soup = BeautifulSoup(html, "html.parser")
            out: List[str] = []
            seen = set()
            # Prefer likely result-title anchors; fall back to broad selection
            for a in soup.select(
                "a.link_tit, a.tit, a.total_tit, a.result_title, a.result-link, a[href]"
            ):
                href = a.get("href")
                url2 = _looks_like_result_link(href)
                if not url2:
                    continue
                if url2 in seen:
                    continue
                seen.add(url2)
                out.append(url2)
                if len(out) >= max_results:
                    break
            return out, False
        except Exception as e:
            logger.debug(f"Naver fetch error: {e}")
            return [], True

    results, had_issue = _fetch_and_parse(params)

    # Soft relax: remove quotes if present and initial results empty
    if not results and (('"' in processed) or ("'" in processed)):
        relaxed = processed.replace('"', "").replace("'", "")
        if relaxed != processed:
            logger.info(
                f"Naver fallback (results empty): removing quotes -> {relaxed}"
            )
            params_relaxed = dict(params)
            params_relaxed["query"] = relaxed
            results_relaxed, had_issue_relaxed = _fetch_and_parse(params_relaxed)
            if results_relaxed:
                results = results_relaxed
            had_issue = had_issue or had_issue_relaxed

    # If still empty and we saw a bot-check/CAPTCHA page, raise to trigger upstream fallback
    if not results and had_issue:
        raise RuntimeError("Naver returned CAPTCHA/bot-check page; no results")

    time.sleep(random.uniform(0.6, 1.2))
    logger.info(f"Naver results: {results}")
    return results

