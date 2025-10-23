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
import random
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import quote_plus, urlparse, parse_qs
from urllib.error import HTTPError

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
from .rate_limiter import get_google_limiter


class SoftBlockError(Exception):
    """Raised when Google returns a soft block / JS challenge page (HTTP 200)."""

    pass


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
    max_attempts = 5
    base_delay = 1.5  # seconds
    limiter = get_google_limiter()

    attempt = 0
    while True:
        # Global pacing to avoid triggering 429 across the app
        limiter.acquire()

        request = Request(url)
        request.add_header("User-Agent", user_agent)
        request.add_header(
            "Accept",
            "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        )
        request.add_header("Accept-Language", "en-US,en;q=0.9")
        request.add_header("Referer", "https://www.google.com/")

        # Add cookies from cookie jar to the request
        cookie_jar.add_cookie_header(request)

        try:
            if verify_ssl:
                response = urlopen(request)
            else:
                context = ssl._create_unverified_context()
                response = urlopen(request, context=context)

            # Extract new cookies from the response and store them
            cookie_jar.extract_cookies(response, request)

            html = response.read()
            response.close()

            # Save cookies to file for persistence
            try:
                cookie_jar.save()
            except Exception:
                pass

            return html

        except HTTPError as e:
            if e.code == 429:
                # Respect Retry-After if provided
                retry_after = (
                    e.headers.get("Retry-After") if hasattr(e, "headers") else None
                )
                if retry_after:
                    try:
                        delay = float(retry_after)
                    except Exception:
                        delay = None
                else:
                    delay = None

                if delay is None:
                    # Exponential backoff with jitter
                    delay = base_delay * (2**attempt) + random.uniform(0.2, 0.8)

                attempt += 1
                if attempt >= max_attempts:
                    logger.error(
                        f"HTTP 429 Too Many Requests after {attempt} attempts; giving up."
                    )
                    raise

                logger.warning(
                    f"HTTP 429 received. Backing off for {delay:.2f}s (attempt {attempt}/{max_attempts})."
                )
                time.sleep(delay)
                continue
            else:
                raise


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

        # Detect soft block / JS challenge pages early and bail out.
        page_text = soup.get_text(" ", strip=True).lower()
        challenge_indicators = (
            "/httpservice/retry/enablejs" in html.decode(errors="ignore").lower()
            or "enable javascript" in page_text
            or "our systems have detected unusual traffic" in page_text
            or ("sorry" in page_text and "automated queries" in page_text)
            or any(
                a.get("href", "").startswith("/httpservice/retry/enablejs")
                or "support.google.com/websearch" in a.get("href", "")
                for a in soup.find_all("a")
            )
        )
        if challenge_indicators:
            logger.warning(
                "Detected Google soft block/JS challenge; aborting cookie search to avoid lock-in."
            )
            raise SoftBlockError("Google soft block / JS challenge detected")
        try:
            anchors = soup.find(id="search").findAll("a")
        except AttributeError:
            gbar = soup.find(id="gbar")
            if gbar:
                gbar.clear()
            anchors = soup.findAll("a")

        # Process every anchored URL.
        valid_found_this_page = 0
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
            valid_found_this_page += 1

            # Increase the results counter. If reached the limit, stop.
            count += 1
            if count >= max_results:
                return results

        # If blocked or no valid results on the first page, stop early.
        if start == 0 and valid_found_this_page == 0:
            logger.warning(
                "No valid results on first page; likely soft-blocked. Stopping pagination."
            )
            break

        # Prepare the URL for the next request. Update start for next page
        start += num

        if num_per_page == 10:
            url = GOOGLE_URL_NEXT_PAGE % vars()
        else:
            url = GOOGLE_URL_NEXT_PAGE_NUM % vars()

        # Sleep between requests. If pause is too small, use safer default with jitter.
        if pause < 1.0:
            base = 3.0
            jitter = random.uniform(0.6, 1.8)
            time.sleep(base + jitter)
        else:
            time.sleep(pause + random.uniform(0.1, 0.6))
    logger.info(f"After cookie_google search, the results are: {results}")
    return results
