import time
import random
import re
from typing import List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from loguru import logger
from urllib.parse import urlparse


LETA_BASE = "https://leta.mullvad.net"
LETA_SEARCH = f"{LETA_BASE}/search"


def preprocess_leta_query(q: str) -> str:
    """Normalize query for Leta (conservative cleanup similar to other engines).

    - Strip leading '*.' after site:
    - Drop positive filetype/html filters (keep negatives)
    - Remove redundant positive site: tokens when a parent domain exists
    - Trim stray boolean operators
    """
    try:
        text = re.sub(r"(?i)(\bsite:\s*)\*\.", r"\1", q)

        def _drop_positive_filetype_html(m: re.Match) -> str:
            sign = m.group("sign") or ""
            return m.group(0) if sign == "-" else ""

        text = re.sub(
            r"(?i)(?P<sign>-?)\b(?:filetype|ext)\s*:\s*(?:html|htm|pdf)\b",
            _drop_positive_filetype_html,
            text,
        )

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


def _looks_like_external(href: str) -> Optional[str]:
    try:
        if not href:
            return None
        if not href.startswith(("http://", "https://")):
            return None
        parsed = urlparse(href)
        if not parsed.netloc:
            return None
        # Exclude Mullvad/Leta internal links
        if "mullvad.net" in parsed.netloc or "leta." in parsed.netloc:
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
    Perform a Leta search and return a list of result URLs.

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
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": LETA_BASE,
    }

    processed = preprocess_leta_query(query)
    # Try common param name `q`; some deployments may use `query` – try both
    param_variants = [
        {"q": processed},
        {"query": processed},
    ]
    logger.info(f"Leta search query: {processed}")

    # Preflight
    try:
        session.get(LETA_BASE, headers=headers, timeout=timeout)
    except Exception as e:
        logger.debug(f"Leta preflight GET failed (continuing): {e}")

    def _fetch_and_parse(q_params: dict) -> Tuple[List[str], bool]:
        try:
            r = session.get(LETA_SEARCH, headers=headers, params=q_params, timeout=timeout)
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
                    "Leta returned a page indicating possible rate limiting or bot check"
                )
                return [], True

            soup = BeautifulSoup(html, "html.parser")
            out: List[str] = []
            seen = set()
            # Heuristic: prefer obvious result anchors if present; otherwise scan all anchors
            for a in soup.select(
                "a.result-title, a.result-header, a[data-testid=\"result-title\"], a[href]"
            ):
                href = a.get("href")
                url2 = _looks_like_external(href)
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
            logger.debug(f"Leta fetch error: {e}")
            return [], True

    results: List[str] = []
    had_issue = False
    # Try both parameter conventions
    for params in param_variants:
        results, had_issue = _fetch_and_parse(params)
        if results:
            break

    # Soft relax: remove quotes if present and initial results empty
    if not results and (('"' in processed) or ("'" in processed)):
        relaxed = processed.replace('"', "").replace("'", "")
        if relaxed != processed:
            logger.info(f"Leta fallback (results empty): removing quotes -> {relaxed}")
            for base_params in param_variants:
                params_relaxed = dict(base_params)
                key = next(iter(params_relaxed.keys()))
                params_relaxed[key] = relaxed
                results_relaxed, had_issue_relaxed = _fetch_and_parse(params_relaxed)
                if results_relaxed:
                    results = results_relaxed
                    break
                had_issue = had_issue or had_issue_relaxed

    # If still empty and we saw a bot-check/CAPTCHA page, raise to trigger upstream fallback
    if not results and had_issue:
        raise RuntimeError("Leta returned CAPTCHA/bot-check page; no results")

    # If still empty, raise to allow upstream to fall back to other engines
    if not results:
        raise RuntimeError("Leta returned no results for the query")

    time.sleep(random.uniform(0.6, 1.2))
    logger.info(f"Leta results: {results}")
    return results


if __name__ == "__main__":
    qs = [
        'site:mozilla.org OR site:w3.org "privacy policy" -filetype:html -ext:htm',
        'site:*.gov cybersecurity roadmap 2025',
    ]
    for q in qs:
        print(q)
        print(preprocess_leta_query(q))

