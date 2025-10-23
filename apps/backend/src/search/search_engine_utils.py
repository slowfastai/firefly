"""
This module contains the URL templates to make Google/Bing/Other search engines searches.
"""

"""
Google URL templates (kept intentionally minimal to reduce bot signals).

Only send `q`, optional `num`, and optional `start`. No `lr`, `tbs`, `cr`, or `safe`.
"""

GOOGLE_URL_HOME = "https://www.google.%(tld)s/"
GOOGLE_URL_SEARCH = "https://www.google.%(tld)s/search?q=%(query)s"
GOOGLE_URL_NEXT_PAGE = "https://www.google.%(tld)s/search?q=%(query)s&start=%(start)d"
GOOGLE_URL_SEARCH_NUM = "https://www.google.%(tld)s/search?q=%(query)s&num=%(num)d"
GOOGLE_URL_NEXT_PAGE_NUM = (
    "https://www.google.%(tld)s/search?q=%(query)s&num=%(num)d&start=%(start)d"
)

# Parameters we may allow in extra_params (kept small)
GOOGLE_URL_PARAMETERS = (
    "q",
    "num",
    "start",
)


# URL templates to make Bing searches.
