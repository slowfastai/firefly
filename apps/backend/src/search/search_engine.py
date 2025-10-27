import re
import sys
import json
import time
import random
import string
import aiohttp
import asyncio
import chardet
import requests
import unicodedata
from io import BytesIO
from pathlib import Path
from typing import Iterable, Set
from urllib.parse import urlparse
from requests.exceptions import Timeout

import langid
import pdfplumber
from tqdm import tqdm
from loguru import logger
from bs4 import BeautifulSoup
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
    LLMConfig,
    LLMExtractionStrategy,
)
from utils.helpers import (
    detect_language_zh_en,
    sent_tokenize_multilingual,
)

from utils.page_content import PageContent, WebPageSummary

# Add the current directory to the Python path for local imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from ._cookie_google_search import search as cookie_google_search
from ._duckduckgo_search import search as duckduckgo_search
from ._startpage_search import search as startpage_search
from ._brave_search import search as brave_search
from ._yandex_search import search as yandex_search
from ._bing_search import search as bing_search
from prompts.prompts_report import get_summarization_instruction

# Initialize Global sync session
global_sync_session = requests.Session()


ERROR_INDICATORS = [
    "limit exceeded",
    "Error fetching",
    "Account balance not enough",
    "Invalid bearer token",
    "HTTP error occurred",
    "Error: Connection error occurred",
    "Error: Request timed out",
    "Unexpected error",
    "Please turn on Javascript",
    "Enable JavaScript",
    "port=443",
    "Please enable cookies",
    "Failed to extract snippet context",
]

INVALID_SEARCH_QUERY = [
    "search_query",
    "search query",
    "your query",
    "my query",
    "your query here",
]


def remove_punctuation(text: str, keep: Iterable[str] | None = None) -> str:
    """
    Remove all Unicode punctuation and symbols (including CJK punctuation and ASCII symbols).
    Use `keep` to specify characters that should not be removed.
    Example: `keep = {"-", "."}` preserves the hyphen and period.

    Notes:
    - Preserves the original characters in output; no width conversion is applied.
    - `keep` matches both the original characters and their NFKC-normalized forms.
    """
    # Build both original and normalized keep sets so we can preserve
    # characters exactly while still recognizing compatibility forms.
    keep_orig: Set[str] = set(keep or [])
    keep_norm: Set[str] = set()
    for item in keep or []:
        normalized = unicodedata.normalize("NFKC", item)
        for ch in normalized:
            keep_norm.add(ch)

    def is_punct_or_symbol(ch: str) -> bool:
        # Remove both punctuation ('P*') and symbol ('S*') categories.
        # P*: Pc, Pd, Pe, Pf, Pi, Po, Ps
        # S*: Sc, Sk, Sm, So (currency, modifier, math, other symbols)
        cat = unicodedata.category(ch)
        return cat.startswith("P") or cat.startswith("S")

    result_chars = []
    for ch in text:
        # Check if this character (or its normalized form) is in keep.
        if (ch in keep_orig) or any(
            c in keep_norm for c in unicodedata.normalize("NFKC", ch)
        ):
            result_chars.append(ch)
            continue
        if not is_punct_or_symbol(ch):
            result_chars.append(ch)
    return "".join(result_chars)


def f1_score(true_set: set, pred_set: set) -> float:
    """Calculate the F1 score between two sets of words."""
    intersection = len(true_set.intersection(pred_set))
    if not intersection:
        return 0.0
    precision = intersection / float(len(pred_set))
    recall = intersection / float(len(true_set))
    return 2 * (precision * recall) / (precision + recall)


def build_llm_strategy_for_crawl4ai(
    llm_name: str,
    llm_base_url: str,
    llm_api_key: str,
    question: str,
    search_query: str,
):
    """
    Build crawl4ai LLM strategy for summarizing web page.

    Args:
        llm_name (str): LLM model name.
        llm_api_key (str): LLM API key.
        llm_base_url (str): LLM base URL.

    Returns:
        LLMExtractionStrategy: LLM extraction strategy.
    """
    # 2) Configure the LLM (LiteLLM-based OpenAI-Compatible Endpoints)
    # Refer to https://docs.litellm.ai/docs/providers/openai_compatible
    #          https://github.com/BerriAI/litellm/issues/8263
    llm_cfg = LLMConfig(
        provider=f"openai/{llm_name}",
        api_token=llm_api_key,
        base_url=llm_base_url,
    )

    return LLMExtractionStrategy(
        llm_config=llm_cfg,
        extraction_type="schema",  # "schema": JSON format; "block": text and small JSON
        schema=WebPageSummary.model_json_schema(),  # Pydantic -> JSON schema
        instruction=get_summarization_instruction(question, search_query),
        input_format="markdown",
        apply_chunking=False,  # Use gemini, no need to chunk
        chunk_token_threshold=1200,
        overlap_rate=0.1,
        # extra_args={
        #     "temperature": 0.7,
        #     "max_tokens": 800,
        # },  # TODO support max_completion_tokens?
        verbose=False,
    )


def extract_snippet_with_context(
    full_text: str, snippet: str, context_chars: int = 3000
) -> tuple[bool, str]:
    """
    Extract the sentence that best matches the snippet and its context from the full text.

    Args:
        full_text (str): The full text extracted from the webpage.
        snippet (str): The snippet/summary to match.
        context_chars (int): Number of characters to include before and after the snippet.

    Returns:
        Tuple[bool, str]: The first element indicates whether extraction was successful, the second element is the extracted context.
    """
    try:
        full_text = full_text[:100000]

        snippet = snippet.lower()
        snippet = remove_punctuation(snippet)
        snippet_words = set(snippet.split())

        best_sentence = None
        best_f1 = 0.2
        sentences = sent_tokenize_multilingual(
            full_text, language=detect_language_zh_en(full_text)
        )

        for sentence in sentences:
            key_sentence = sentence.lower()
            key_sentence = remove_punctuation(key_sentence)
            sentence_words = set(key_sentence.split())
            f1 = f1_score(snippet_words, sentence_words)
            if f1 > best_f1:
                best_f1 = f1
                best_sentence = sentence

        if best_sentence:
            para_start = full_text.find(best_sentence)
            para_end = para_start + len(best_sentence)
            start_index = max(0, para_start - context_chars)
            end_index = min(len(full_text), para_end + context_chars)
            context = full_text[start_index:end_index]
            return True, context
        else:
            # If no matching sentence is found, return the first context_chars*2 characters of the full text
            return False, full_text[: context_chars * 2]
    except Exception as e:
        logger.error(
            f"Error (extract_snippet_with_context): Failed to extract snippet context due to {str(e)}"
        )
        return False, f"Failed to extract snippet context due to {str(e)}"


def extract_relevant_info(search_results):
    """
    Extract relevant information from Bing search results.

    Args:
        search_results (dict): JSON response from the Bing Web Search API.

    Returns:
        list: A list of dictionaries containing the extracted information.
    """
    useful_info = []
    # webPages contains top k urls and snippets
    if "webPages" in search_results and "value" in search_results["webPages"]:
        for id, result in enumerate(search_results["webPages"]["value"]):
            info = {
                "id": id + 1,  # Increment id for easier subsequent operations
                "title": result.get("name", ""),  # page title
                "url": result.get("url", ""),  # page url
                "site_name": result.get("siteName", ""),  # site name
                "date": result.get("datePublished", "").split("T")[
                    0
                ],  # page publish date
                "snippet": result.get(
                    "snippet", ""
                ),  # short description of page content, important
                # Add context content to the information
                "context": "",  # Reserved field to be filled later
            }
            useful_info.append(info)

    return useful_info


class RateLimiter:
    def __init__(self, rate_limit: int, time_window: int = 60):
        """
        Initialize rate limiter

        Args:
            rate_limit: Maximum number of requests allowed in the time window
            time_window: Time window size in seconds, default 60 seconds
        """
        self.rate_limit = rate_limit
        self.time_window = time_window
        self.tokens = rate_limit
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Acquire a token, wait if no tokens are available"""
        async with self.lock:
            while self.tokens <= 0:
                now = time.time()
                time_passed = now - self.last_update
                self.tokens = min(
                    self.rate_limit,
                    self.tokens + (time_passed * self.rate_limit / self.time_window),
                )
                self.last_update = now
                if self.tokens <= 0:
                    await asyncio.sleep(
                        random.randint(5, 30)
                    )  # Wait for random seconds before retry

            self.tokens -= 1
            return True


# Create global rate limiter instance
jina_rate_limiter = RateLimiter(
    rate_limit=130
)  # 130 requests per minute to avoid errors


async def extract_text_from_url_async(
    url: str,
    async_session: aiohttp.ClientSession,
    extractor: str = "requests",
    extractor_kwargs: dict = {},
    snippet: str | None = None,
    keep_links: bool = False,
) -> str:
    """
    Async version of extraction web content from url
    Extract text from a URL. If a snippet is provided, extract the context related to it.

    Args:
        url (str): The URL to extract text from.
        extractor (str): The extractor to use, ["requests", "jina"]
        extractor_kwargs (Optional[dict]): The kwargs for the extractor.
            {"user_agent": str, "jina_api_key": str, "}
        snippet (Optional[str]): The snippet to search for.
        keep_links (bool): Whether to keep links in the extracted text.

    Returns:
        str: Extracted text or context.
    """
    try:
        if extractor == "jina":
            await jina_rate_limiter.acquire()

            jina_headers = {
                "Authorization": f"Bearer {extractor_kwargs['jina_api_key']}",
                "X-Return-Format": "text",
            }

            async with async_session.get(
                f"https://r.jina.ai/{url}", headers=jina_headers
            ) as response:
                text = await response.text()
                if not keep_links:
                    pattern = r"\(https?:.*?\)|\[https?:.*?\]"
                    text = re.sub(pattern, "", text)
                text = (
                    text.replace("---", "-")
                    .replace("===", "=")
                    .replace("   ", " ")
                    .replace("   ", " ")
                )
        # elif extractor == "crawl4ai":
        #     browser_conf = BrowserConfig(headless=True, user_agent=extractor_kwargs.get("user_agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"))  # or False to see the browser
        #     run_conf = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
        #     async with AsyncWebCrawler(config=browser_conf) as crawler:
        #         result = await crawler.arun(
        #             url=url,
        #             config=run_conf
        #         )
        #         if result.success:
        #             text = result.markdown.raw_markdown
        #             if not keep_links:
        #                 pattern = r"\(https?:.*?\)|\[https?:.*?\]"
        #                 text = re.sub(pattern, "", text)
        #         else:
        #             logger.error(f"[Error] Crawl4AI failed to fetch {url}")
        elif extractor == "requests":
            if "pdf" in url:
                text = await extract_pdf_text_async(url, async_session)
                return text[:10000]

            async with async_session.get(url) as response:
                # Detect and handle encoding
                content_type = response.headers.get("content-type", "").lower()
                if "charset" in content_type:
                    charset = content_type.split("charset=")[-1]
                    html = await response.text(encoding=charset)
                else:
                    # If no encoding is specified, read content as bytes first
                    content = await response.read()
                    # Use chardet to detect encoding
                    detected = chardet.detect(content)
                    encoding = detected["encoding"] if detected["encoding"] else "utf-8"
                    html = content.decode(encoding, errors="replace")

                # Check for error indicators
                has_error = (
                    (
                        any(
                            indicator.lower() in html.lower()
                            for indicator in ERROR_INDICATORS
                        )
                        and len(html.split()) < 64
                    )
                    or len(html) < 50
                    or len(html.split()) < 20
                )
                if has_error:
                    return (
                        f"[Error] error: Content too short or contains error indicators"
                    )

                try:
                    soup = BeautifulSoup(html, "lxml")
                except Exception:
                    soup = BeautifulSoup(html, "html.parser")

                if keep_links:
                    # Similar link handling logic as in synchronous version
                    for element in soup.find_all(["script", "style", "meta", "link"]):
                        element.decompose()

                    text_parts = []
                    for element in (
                        soup.body.descendants if soup.body else soup.descendants
                    ):
                        if isinstance(element, str) and element.strip():
                            cleaned_text = " ".join(element.strip().split())
                            if cleaned_text:
                                text_parts.append(cleaned_text)
                        elif element.name == "a" and element.get("href"):
                            href = element.get("href")
                            link_text = element.get_text(strip=True)
                            if href and link_text:
                                if href.startswith("/"):
                                    base_url = "/".join(url.split("/")[:3])
                                    href = base_url + href
                                elif not href.startswith(("http://", "https://")):
                                    href = url.rstrip("/") + "/" + href
                                text_parts.append(f"[{link_text}]({href})")

                    text = " ".join(text_parts)
                    text = " ".join(text.split())
                else:
                    text = soup.get_text(separator=" ", strip=True)

        if snippet:
            success, context = extract_snippet_with_context(text, snippet)
            return context if success else text
        else:
            return text[:50000]

    except Exception as e:
        error_message = f"[Error] Error fetching {url}: {str(e)}"
        logger.error(error_message)
        return error_message


async def fetch_page_content_async(
    urls: list[str],
    extractor: str = "crawl4ai",
    extractor_kwargs: dict = {},
    snippets: dict[str, str] | None = None,
    show_progress: bool = False,
    keep_links: bool = False,
    max_concurrent: int = 16,
) -> dict[str, str]:
    """
    Asynchronously fetch content from multiple urls.

    Args:
        urls (list):  List of URLs to fetch.
        extractor (str):  The extractor to use for fetching content.
        extractor_kwargs (dict):  Keyword arguments for the extractor.
        snippets (dict):
        show_progress (bool):
        keep_links (bool):
        max_concurrent (int):

    return: A dictionary mapping URLs to their extracted content (PageContent format).
    """
    headers = {
        "User-Agent": extractor_kwargs.get(
            "user_agent",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        ),
        "Referer": "https://www.google.com/",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",  # ALWAYS USE ENGLISH to search
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    async def process_urls():
        if extractor == "crawl4ai":
            browser_conf = BrowserConfig(
                # Config browser is headless
                headless=True,
                user_agent=headers["User-Agent"],
            )
            llm_strategy = build_llm_strategy_for_crawl4ai(
                llm_name=extractor_kwargs["llm_name"],
                llm_base_url=extractor_kwargs["llm_base_url"],
                llm_api_key=extractor_kwargs["llm_api_key"],
                question=extractor_kwargs["question"],
                search_query=extractor_kwargs["search_query"],
            )
            run_conf = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                extraction_strategy=llm_strategy,
                stream=False,
            )
            async with AsyncWebCrawler(config=browser_conf) as crawler:
                # Using the async_run_many method to run the crawler
                results = await crawler.arun_many(urls=urls, config=run_conf)
                url2content = {}
                for res in results:
                    if res.success:
                        # TODO: raw_markdown retains many irrelevant HTML tags. Should we enable a content filter in crawl4ai?
                        # It’s possible that later WebThinker processing (extracting snippet context) won’t keep raw_content anyway.
                        text = res.markdown.raw_markdown
                        if not keep_links:
                            pattern = r"\(https?:.*?\)|\[https?:.*?\]"
                            text = re.sub(pattern, "", text)
                        # logger.info(f"type(res.extracted_content): {type(res.extracted_content)}")  # str
                        # logger.info(f"res.extracted_content: {res.extracted_content}")
                        try:
                            extracted_content = json.loads(res.extracted_content)
                            if isinstance(extracted_content, list):
                                websummary = (
                                    extracted_content[0] if extracted_content else {}
                                )
                            elif isinstance(extracted_content, dict):
                                websummary = extracted_content
                            else:
                                websummary = {}
                        except json.JSONDecodeError:
                            logger.error(
                                f"[ERROR] crawl4ai extract summarization from {res.url}, \
                                    res.extracted_content: {res.extracted_content}"
                            )
                            # websummary = {"summary": res.extracted_content}
                            websummary = {}

                        # Extract site_name from url
                        parsed_url = urlparse(res.url)
                        site_name = parsed_url.netloc

                        # Construct PageContent object
                        url_content = PageContent(
                            url=res.url,
                            raw_content=text,
                            ranking_id=-1,
                            site_name=site_name,
                            title=websummary.get("title", None),
                            snippet=websummary.get("summary", None),
                            key_points=websummary.get("key_points", [""]),
                        )
                        url2content[res.url] = url_content
                    else:
                        logger.error(
                            f"[Error] Crawl4AI Error fetching {res.url}: {res.error_message}"
                        )
                return url2content
        else:  # "requests" or "jina"
            connector = aiohttp.TCPConnector(limit=max_concurrent)
            timeout = aiohttp.ClientTimeout(total=120)  # TODO how to set timeout?
            async with aiohttp.ClientSession(
                connector=connector, timeout=timeout, headers=headers
            ) as async_session:
                tasks = []
                for url in urls:
                    task = extract_text_from_url_async(
                        url,
                        async_session,
                        extractor=extractor,
                        extractor_kwargs=extractor_kwargs,
                        snippet=snippets.get(url, None) if snippets else None,
                        keep_links=keep_links,
                    )
                    tasks.append(task)

                if show_progress:
                    results = []
                    for task in tqdm(
                        asyncio.as_completed(tasks),
                        total=len(tasks),
                        desc="Fetching URLs",
                    ):
                        result = await task
                        results.append(result)
                else:
                    results = await asyncio.gather(*tasks)

                return {
                    url: PageContent(
                        url=url,
                        raw_content=result,
                        ranking_id=-1,
                    )
                    for url, result in zip(urls, results)
                }

    return await process_urls()


async def extract_pdf_text_async(url: str, session: aiohttp.ClientSession) -> str:
    """
    Asynchronously extract text from a PDF.

    Args:
        url (str): URL of the PDF file.
        session (aiohttp.ClientSession): Aiohttp client session.

    Returns:
        str: Extracted text content or error message.
    """
    try:
        timeout = aiohttp.ClientTimeout(total=20)
        async with session.get(url, timeout=timeout) as response:
            if response.status != 200:
                error_message = f"[Error](extract_pdf_text_async) Unable to retrieve the PDF (status code {response.status}) from {url}"
                logger.error(error_message)
                return error_message

            content = await response.read()

            # Open the PDF file using pdfplumber
            with pdfplumber.open(BytesIO(content)) as pdf:
                full_text = ""
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text

            # Limit the text length
            cleaned_text = full_text
            return cleaned_text
    except asyncio.TimeoutError:
        error_message = (
            f"[Error](extract_pdf_text_async) Request timed out after 20 seconds"
        )
        logger.error(error_message)
        return error_message
    except Exception as e:
        error_message = (
            f"[Error](extract_pdf_text_async) Error extracting text from PDF: {str(e)}"
        )
        logger.error(error_message)
        return error_message


def google_serper_search(query: str, api_key: str, timeout: int = 20):
    """
    Perform a search using the Google Serper API.

    Args:
        query (str): Search query.
        api_key (str): API key for Google Serper API.
        timeout (int or float or tuple): Request timeout in seconds.

    Returns:
        dict: JSON response of the search results. Returns empty dict if request fails.
    """
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            response = requests.post(
                url, headers=headers, data=payload, timeout=timeout
            )
            response.raise_for_status()  # Raise exception if the request failed
            search_results = response.json()
            return search_results
        except Timeout:
            retry_count += 1
            if retry_count == max_retries:
                print(
                    f"Google Serper API request timed out ({timeout} seconds) for query: {query} after {max_retries} retries"
                )
                return {}
            print(
                f"Google Serper API Timeout occurred, retrying ({retry_count}/{max_retries})..."
            )
        except requests.exceptions.RequestException as e:
            retry_count += 1
            if retry_count == max_retries:
                print(
                    f"Google Serper API Request Error occurred: {e} after {max_retries} retries"
                )
                return {}
            print(
                f"Google Serper API Request Error occurred, retrying ({retry_count}/{max_retries})..."
            )
        time.sleep(1)  # Wait 1 second between retries

    return {}
