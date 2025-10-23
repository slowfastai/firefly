import time, random, urllib.parse, requests
from bs4 import BeautifulSoup

DDG_HTML = "https://html.duckduckgo.com/html/"

def search_ddg(query: str, max_results: int = 10, offset: int = 0):
    session = requests.Session()
    headers = {
        # 更“真实”的 UA；有些返回会对 UA 比较敏感
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": DDG_HTML,
    }

    # 1) 先 GET 一次拿到 cookie（有时直接 POST 会被空页/风控）
    session.get(DDG_HTML, headers=headers, timeout=20)

    # 2) 用 POST 提交查询（DDG HTML 端支持 POST），带上偏移量 s（分页）
    data = {
        "q": query,
        "s": str(offset),  # 分页偏移：0, 30, 60...
        "ia": "web",
    }
    r = session.post(DDG_HTML, headers=headers, data=data, timeout=20)
    r.raise_for_status()

    html = r.text

    # 3) 简单的风控/空页检测，方便你定位问题
    lowered = html.lower()
    if ("captcha" in lowered) or ("unusual traffic" in lowered) or ("verify you are a human" in lowered):
        # 返回空数组，但给出可定位信息
        return []

    soup = BeautifulSoup(html, "html.parser")

    results = []

    # 4) 更鲁邦的选择器：先按常见类名拿；拿不到再退化
    # 常规（html 端）结果：div.result > a.result__a
    for a in soup.select("div.result a.result__a"):
        url = a.get("href")
        title = a.get_text(" ", strip=True)
        if url and title:
            results.append({"title": title, "url": url})
        if len(results) >= max_results:
            break

    # 5) 退化选择器（如果 DDG 改了类名，仍有机会抓到）
    if not results:
        for a in soup.select("#links a[href]"):
            # 过滤掉内链/跳转锚点
            href = a.get("href")
            title = a.get_text(" ", strip=True)
            if not href or not title:
                continue
            if href.startswith("/") or href.startswith("#"):
                continue
            # 避免抓导航/无标题链接
            if len(title) < 2:
                continue
            results.append({"title": title, "url": href})
            if len(results) >= max_results:
                break

    # 6) 随机延时，降低被限流概率
    time.sleep(random.uniform(0.6, 1.2))
    return results


print(search_ddg("Who is the best football player in the world?"))
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

print(search_ddg('"how the Sun formed" "nebular hypothesis" "gravitational collapse" solar system formation NASA'))