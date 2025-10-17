"""
This module contains the URL templates to make Google/Bing/Other search engines searches.
"""

# URL templates to make Google searches.
GOOGLE_URL_HOME = "https://www.google.%(tld)s/"
GOOGLE_URL_SEARCH = (
    "https://www.google.%(tld)s/search?lr=lang_%(lang)s&"
    "q=%(query)s&btnG=Google+Search&tbs=%(tbs)s&safe=%(safe)s&"
    "cr=%(country)s&filter=0"
)
GOOGLE_URL_NEXT_PAGE = (
    "https://www.google.%(tld)s/search?lr=lang_%(lang)s&"
    "q=%(query)s&start=%(start)d&tbs=%(tbs)s&safe=%(safe)s&"
    "cr=%(country)s&filter=0"
)
GOOGLE_URL_SEARCH_NUM = (
    "https://www.google.%(tld)s/search?lr=lang_%(lang)s&"
    "q=%(query)s&num=%(num)d&btnG=Google+Search&tbs=%(tbs)s"
    "&safe=%(safe)s&cr=%(country)s&filter=0"
)
GOOGLE_URL_NEXT_PAGE_NUM = (
    "https://www.google.%(tld)s/search?lr=lang_%(lang)s&"
    "q=%(query)s&num=%(num)d&start=%(start)d&tbs=%(tbs)s"
    "&safe=%(safe)s&cr=%(country)s&filter=0"
)

# Parameters available in Google search URLs:
# hl: interface language (host language)
# q: search query
# num: number of results per page
# btnG: search button identifier
# start: starting result index
# tbs: time-based search restriction
# safe: SafeSearch setting
# cr: country or region
# filter: duplicate results filter
GOOGLE_URL_PARAMETERS = (
    "hl",
    "q",
    "num",
    "btnG",
    "start",
    "tbs",
    "safe",
    "cr",
    "filter",
)


# URL templates to make Bing searches.
