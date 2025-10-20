"""
This script is a modified version of the original googlesearch script.
It uses the cookie jar to perform the search.

The original script is from:
    https://github.com/MarioVilas/googlesearch/blob/master/googlesearch/__init__.py

The original script is licensed under the BSD 3-Clause License.
"""

import ssl
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import quote_plus, urlparse, parse_qs

from loguru import logger
from bs4 import BeautifulSoup

# Add the src directory to the Python path for imports
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from utils.user_agent import get_browser_user_agent
from browser_cookies import load_cookies
from browser_cookies import DeepResearchCookieJar
from search_engine_utils import (
    GOOGLE_URL_SEARCH,
    GOOGLE_URL_NEXT_PAGE,
    GOOGLE_URL_SEARCH_NUM,
    GOOGLE_URL_NEXT_PAGE_NUM,
    GOOGLE_URL_PARAMETERS,
    # GOOGLE_URL_HOME,
)


def get_page(
    url: str,
    user_agent: str,
    cookie_jar: DeepResearchCookieJar,
    verify_ssl: bool = True,
):
    """
    Request the given URL and return the response page, using the cookie jar.

    :param str url: URL to retrieve.
    :param str user_agent: User agent for the HTTP requests.
    :param DeepResearchCookieJar cookie_jar: Cookie jar to use for session management.
    :param bool verify_ssl: Verify the SSL certificate to prevent
        traffic interception attacks. Defaults to True.

    :rtype: str
    :return: Web page retrieved for the given URL.

    :raises IOError: An exception is raised on error.
    :raises urllib2.URLError: An exception is raised on error.
    :raises urllib2.HTTPError: An exception is raised on error.
    """
    request = Request(url)
    request.add_header("User-Agent", user_agent)

    # Add cookies from cookie jar to the request
    # This makes the request look more like a real browser
    cookie_jar.add_cookie_header(request)

    if verify_ssl:
        response = urlopen(request)
    else:
        context = ssl._create_unverified_context()
        response = urlopen(request, context=context)

    # Extract new cookies from the response and store them
    # This maintains session state across requests
    cookie_jar.extract_cookies(response, request)

    html = response.read()
    response.close()

    # Save cookies to file for persistence
    # This allows cookies to be reused in future sessions
    try:
        cookie_jar.save()
    except Exception:
        pass

    return html


def filter_search_result(link: str, include_google_links: bool = False) -> str | None:
    """
    Filter links found in the Google result pages HTML code.
    Returns None if the link doesn't yield a valid result.

    Args:
        link: URL link to filter.
        include_google_links: Includes links pointing to a Google domain.

    Returns:
        Filtered link or None if the link is invalid.
    """
    try:
        # Decode hidden URLs.
        if link.startswith(
            "/url?"
        ):  # If the link starts with '/url?', extract the 'q' parameter from the query string
            o = urlparse(link, "http")
            link = parse_qs(o.query).get("q")[0]

        o = urlparse(link, "http")

        # Check if the link is an absolute URL.
        if not o.netloc:
            return None

        # If excluding Google links, return None if 'google' is in the domain.
        if not include_google_links and "google" in o.netloc:
            return None

        return link

    except Exception as e:
        logger.error(f"Error filtering result: {e}")
        return None


def search(
    query: str,
    user_agent: str | None = None,
    cookie_jar: DeepResearchCookieJar | None = None,
    num_per_page: int = 10,
    max_results: int = 10,
    pause: float = 0.5,
    tld: str = "com",
    lang: str = "en",
    tbs: str = "0",
    safe: str = "off",
    country: str = "US",
    extra_params: dict | None = None,
    verify_ssl: bool = True,
    include_google_links: bool = False,
):
    """
    Search the given query string using Google.

    Args:
        query: Query string. Must NOT be url-encoded.
        tld: Top level domain.
        lang: Language.
        tbs: Time limits (i.e "qdr:h" => last hour,
            "qdr:d" => last 24 hours, "qdr:m" => last month).
        safe: Safe search.
        num_per_page: Number of results per page.
        user_agent: User agent for the HTTP requests.
        cookie_jar: Cookie jar to use for session management.
        max_results: Maximum number of results to retrieve.
        pause: Lapse to wait between HTTP requests, measured in seconds.
        country: Country or region to focus the search on. Similar to
            changing the TLD, but does not yield exactly the same results.
        extra_params: A dictionary of extra HTTP GET
            parameters, which must be URL encoded. For example if you don't want
            Google to filter similar results you can set the extra_params to
            {'filter': '0'} which will append '&filter=0' to every query.
        verify_ssl: Verify the SSL certificate to prevent
            traffic interception attacks. Defaults to True.
        include_google_links: Includes links pointing to a Google domain.

    Returns:
        List of found URLs.
    """
    if user_agent is None:
        logger.info(
            "User agent not specified. Automatically extracting from Chrome browser. "
            "This may differ from your current browser settings. "
            "To use a specific user agent, pass it as a parameter."
        )
        user_agent = get_browser_user_agent("chrome")
    if cookie_jar is None:
        logger.info(
            "Cookie jar not specified. Automatically extracting cookies from Chrome browser. "
            "This may differ from your current browser settings. "
            "To use specific cookies, pass a cookie jar as a parameter."
        )
        cookie_jar = load_cookies()

    hashes = set()
    count = 0
    results = []  # search results, list of url

    # Initialize pagination variables
    num = num_per_page
    start = 0

    # Prepare the query string.
    logger.info(f"Before quote_plus, the query is: {query}")
    query = quote_plus(query)
    logger.info(f"After quote_plus, the query is: {query}")

    if not extra_params:
        extra_params = {}

    # Check extra_params for overlapping.
    for builtin_param in GOOGLE_URL_PARAMETERS:
        if builtin_param in extra_params.keys():
            logger.error(
                f'GET parameter "{builtin_param}" is overlapping with \
                the built-in GET parameter'
            )
            raise ValueError(
                f'GET parameter "{builtin_param}" is overlapping with \
                the built-in GET parameter'
            )

    # Get the cookie from the Google home page.
    if num_per_page == 10:
        url = GOOGLE_URL_SEARCH % vars()
    else:
        url = GOOGLE_URL_SEARCH_NUM % vars()
    logger.info(f"The url is: {url}")
    while count < max_results:
        for k, v in extra_params.items():
            k = quote_plus(k)
            v = quote_plus(v)
            url = url + ("&%s=%s" % (k, v))
        logger.info(f"After concatenating extra_params, the url is: {url}")
        # Request the Google Search results page.
        html = get_page(url, user_agent, cookie_jar, verify_ssl)

        # save html to file
        # with open("google_search.html", "w") as f:
        #     f.write(html.decode("utf-8"))

        # Parse the response and get every anchored URL.
        soup = BeautifulSoup(html, "html.parser")
        try:
            anchors = soup.find(id="search").findAll("a")
        except AttributeError:
            gbar = soup.find(id="gbar")
            if gbar:
                gbar.clear()
            anchors = soup.findAll("a")

        # Process every anchored URL.
        for a in anchors:
            try:  # Get the URL from the anchor tag.
                link = a["href"]
                logger.info(f"The link is: {link}")
            except KeyError:
                continue

            # Filter invalid links and links pointing to Google itself.
            link = filter_search_result(link, include_google_links)
            logger.info(f"The filtered link is: {link}")
            if not link:
                continue

            # Discard repeated results.
            h = hash(link)
            if h in hashes:
                continue
            hashes.add(h)
            results.append(link)

            # Increase the results counter. If reached the limit, stop.
            count += 1
            if count >= max_results:
                return results

        # Prepare the URL for the next request. Update start for next page
        start += num

        if num_per_page == 10:
            url = GOOGLE_URL_NEXT_PAGE % vars()
        else:
            url = GOOGLE_URL_NEXT_PAGE_NUM % vars()

        # Sleep between requests.
        time.sleep(pause)
    logger.info(f"After cookie_google search, the results are: {results}")
    return results
