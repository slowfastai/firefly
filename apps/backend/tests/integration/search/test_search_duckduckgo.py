import time, random, urllib.parse, requests
from bs4 import BeautifulSoup

DDG_HTML = "https://html.duckduckgo.com/html/"


def search_ddg(query: str, max_results: int = 10, offset: int = 0):
    session = requests.Session()
    headers = {
        # A more realistic UA; some responses are UA-sensitive
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": DDG_HTML,
    }

    # 1) Initial GET to obtain cookies (direct POST may yield empty/blocked pages)
    session.get(DDG_HTML, headers=headers, timeout=20)

    # 2) Submit the query via POST (HTML endpoint supports POST); include s for pagination
    data = {
        "q": query,
        "s": str(offset),  # page offset: 0, 30, 60...
        "ia": "web",
    }
    r = session.post(DDG_HTML, headers=headers, data=data, timeout=20)
    r.raise_for_status()

    html = r.text

    # 3) Simple bot/blank-page detection to aid debugging
    lowered = html.lower()
    if (
        ("captcha" in lowered)
        or ("unusual traffic" in lowered)
        or ("verify you are a human" in lowered)
    ):
        # Return empty results while signaling a likely rate-limit/verification page
        return []

    soup = BeautifulSoup(html, "html.parser")

    results = []

    # 4) More robust selection: prefer common class names; fall back if needed
    # Typical (HTML endpoint) results: div.result > a.result__a
    for a in soup.select("div.result a.result__a"):
        url = a.get("href")
        title = a.get_text(" ", strip=True)
        if url and title:
            results.append({"title": title, "url": url})
        if len(results) >= max_results:
            break

    # 5) Fallback selector (if DDG changes class names, still try to extract)
    if not results:
        for a in soup.select("#links a[href]"):
            # Filter internal links/anchors
            href = a.get("href")
            title = a.get_text(" ", strip=True)
            if not href or not title:
                continue
            if href.startswith("/") or href.startswith("#"):
                continue
            # Skip nav or title-less links
            if len(title) < 2:
                continue
            results.append({"title": title, "url": href})
            if len(results) >= max_results:
                break

    # 6) Random delay to reduce rate-limit likelihood
    time.sleep(random.uniform(0.6, 1.2))
    return results


# print(search_ddg("Who is the best football player in the world?"))
"""
[
    {
        'title': 'Best Player In The World 2025: Top 10 Football Players', 
        'url': 'https://www.sportsdunia.com/football-analysis/top-10-best-football-players-in-the-world'
    }, 
    {
        'title': '30 Best Footballers in the World (2025) - GiveMeSport', 
        'url': 'https://www.givemesport.com/best-football-players-in-the-world/'
    },
    ...
]
"""

# print(search_ddg('"how the Sun formed" "nebular hypothesis" "gravitational collapse" solar system formation NASA'))
