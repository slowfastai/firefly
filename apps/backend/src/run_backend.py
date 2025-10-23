import os
import re
import sys
import json
import time
import asyncio
import argparse
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Callable, Any, Optional, Mapping, Union

# Add the src directory to the Python path
src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))

import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi


from search.search_engine import (
    extract_snippet_with_context,
    fetch_page_content_async,
    cookie_google_search,
    duckduckgo_search,
    google_serper_search,
    ERROR_INDICATORS,
    INVALID_SEARCH_QUERY,
)
from prompts.prompts_report import (
    get_click_intent_instruction,
    get_report_webthinker_instruction,
    get_search_plan_instruction,
    get_deep_web_explorer_instruction,
    get_write_section_instruction,
    get_edit_article_instruction,
    get_title_instruction,
    get_click_web_page_reader_instruction,
    get_final_report_instruction,
    get_detailed_web_page_reader_instruction,
    get_query_clarification_instruction,
    get_query_rewriting_instruction,
)

from utils.user_agent import get_browser_user_agent
from utils.helpers import SQLiteCache, detect_language_zh_en, word_tokenize_multilingual
from browser_cookies import load_cookies, DeepResearchCookieJar
from utils.report_cli import parse_args, get_logger, set_seed, LLMResponse
from utils.async_llm_manager import create_llm_client, LLMClient

# Load environment from repo root .env and backend/.env for reliability
load_dotenv()  # standard .env in CWD / parents, if present
backend_env = Path(__file__).resolve().parents[1] / ".env"
if backend_env.exists():
    load_dotenv(backend_env, override=False)


# ---- Cooperative cancel helpers (desktop integration) ----
def _cancel_file_path() -> Optional[str]:
    return os.environ.get("DR_CANCEL_FILE")


def _is_cancelled() -> bool:
    cf = _cancel_file_path()
    return bool(cf and os.path.exists(cf))


async def _check_cancelled(emit: Optional[Callable[[dict], None]] = None):
    """Raise CancelledError if a cancel flag file exists.

    The Electron desktop sets DR_CANCEL_FILE per session and toggles it on cancel.
    This provides a cooperative early-exit path in addition to OS signals.
    """
    if _is_cancelled():
        try:
            if emit:
                emit({"type": "cancelled", "payload": {"message": "Cancelled by flag"}})
        finally:
            raise asyncio.CancelledError()


# Define special tokens
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"

BEGIN_CLICK_LINK = "<|begin_click_link|>"
END_CLICK_LINK = "<|end_click_link|>"
BEGIN_CLICK_RESULT = "<|begin_click_result|>"
END_CLICK_RESULT = "<|end_click_result|>"

BEGIN_WRITE_SECTION = "<|begin_write_section|>"
END_WRITE_SECTION = "<|end_write_section|>"
BEGIN_EDIT_ARTICLE = "<|begin_edit_article|>"
END_EDIT_ARTICLE = "<|end_edit_article|>"
BEGIN_CHECK_ARTICLE = "<|begin_check_article|>"
END_CHECK_ARTICLE = "<|end_check_article|>"
BEGIN_CHECK_RESULT = "<|begin_article_outline|>"
END_CHECK_RESULT = "<|end_article_outline|>"

BEGIN_THINK = "<|begin_think|>"
END_THINK = "<|end_think|>"

REASONING_TRAJECTORY_MAX_TOKENS = 100000  # TODO What is an appropriate value?
MAX_INTERACTIONS = 80  # Maximum number of total interactions(reasoning steps)
DEEP_WEB_EXPLORE_MAX_TOKENS = 20000
DEEP_WEB_EXPLORE_MAX_INTERACTIONS = 10  # Maximum combined number of searches and clicks


def normalize_instruction_following_response(
    resp: str, begin_marker: str, end_marker: str
) -> str:
    """
    Normalize an LLM response to the canonical search-query template:
      <|begin_search_query|>{content}<|end_search_query|>

    - Accepts minor tag deviations like:
        </|end_search_query|>, <|end_search_query| (missing '>'),
        mixed-case, extra spaces, etc.
    - If only start exists → take text after it.
    - If only end exists → take text before it.
    - If neither exists → wrap the whole text.
    - Collapses internal whitespace to a single space.
    """
    s = resp.strip()

    # 1) Canonicalize tag variants (tolerant to spaces, optional '/', missing '>', case)
    s = re.sub(r"<\s*\|?\s*begin_search_query\s*\|\s*>", begin_marker, s, flags=re.I)
    s = re.sub(r"<\s*/?\s*\|?\s*end_search_query\s*\|\s*>", end_marker, s, flags=re.I)

    # 2) Find first canonical begin / end
    b = s.find(begin_marker)
    e = s.find(end_marker)

    if b != -1 and (e != -1) and (b < e):
        content = s[b + len(begin_marker) : e]
    elif b != -1:
        content = s[b + len(end_marker) :]
    elif e != -1:
        content = s[:e]
    else:
        content = s

    # 3) Normalize whitespace inside content
    content = re.sub(r"\s+", " ", content).strip()

    # 4) Return canonical wrapped form
    return f"{begin_marker}{content}{end_marker}"


def normalize_tagged_text(
    resp: str,
    begin_marker: str | None = None,
    end_marker: str | None = None,
    collapse_ws: bool = True,
) -> str:
    """
    Unified handler that:
      1. Normalizes begin/end tag variants (case differences, missing '>', etc.)
      2. Fills in missing paired tags (behavior configured via TAG_PAIRS)
      3. When `begin_marker`/`end_marker` are provided, extracts and returns the canonical block (compatible with the previous normalize helper)
    """
    TAG_PAIRS = (
        (
            BEGIN_THINK,
            END_THINK,
            True,
        ),  # True means prepend the end tag if it's missing
        (BEGIN_SEARCH_QUERY, END_SEARCH_QUERY, False),
        (BEGIN_WRITE_SECTION, END_WRITE_SECTION, False),
        (BEGIN_EDIT_ARTICLE, END_EDIT_ARTICLE, False),
        (BEGIN_CHECK_RESULT, END_CHECK_RESULT, False),
        (BEGIN_CLICK_LINK, END_CLICK_LINK, False),
    )
    text = resp.strip()

    # Step 1: canonicalize all known tags
    for begin, end, prepend_end in TAG_PAIRS:
        # Fix end-tag variants such as "<|end_search_query|" or "</|end_search_query|"
        text = re.sub(rf"<\s*/?\s*\|?\s*{end[2:-2]}\s*\|?\s*>", end, text, flags=re.I)
        text = re.sub(rf"<\s*\|?\s*{begin[2:-2]}\s*\|?\s*>", begin, text, flags=re.I)

        # Step 2: add a missing end tag if needed
        if text.count(begin) and end not in text:
            text = f"{end}{text}" if prepend_end else f"{text}{end}"

    # If begin/end markers are provided, fall back to the original extraction behavior
    if begin_marker and end_marker:
        start = text.find(begin_marker)
        end_pos = text.find(end_marker)
        if start != -1 and end_pos != -1 and start < end_pos:
            content = text[start + len(begin_marker) : end_pos]
        elif start != -1:
            content = text[start + len(begin_marker) :]
        elif end_pos != -1:
            content = text[:end_pos]
        else:
            content = text

        if collapse_ws:
            content = re.sub(r"\s+", " ", content).strip()
        return f"{begin_marker}{content}{end_marker}"

    return text


def extract_token_usage(model_response: LLMResponse):
    """
    Extract token usage from a model response.

    Args:
        model_response (LLMResponse): The model response object.
    Returns:
            prompt_tokens (int): The number of tokens used for prompting.
            response_tokens (int): The number of tokens used for response, exluding reasoning tokens.
    """
    usage = model_response.response_metadata.get("usage")
    prompt_tokens = completion_tokens = reasoning_tokens = response_tokens = 0
    if usage:
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)
        details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = (
            (getattr(details, "reasoning_tokens", 0) or 0) if details else 0
        )
        response_tokens = completion_tokens - reasoning_tokens
    return prompt_tokens, response_tokens


def extract_between(text: str, start_marker: str, end_marker: str, strip: bool = True):
    """
    Extract a substring from `text` using optional start and end markers.

    Rules:
    1. If both `start_marker` and `end_marker` exist in the correct order,
       return the text between them.
    2. If only `start_marker` exists, return everything after it.
    3. If only `end_marker` exists, return everything before it.
    4. If neither marker exists, return None.

    Args:
        text (str): The input string to search in.
        start_marker (str): The starting delimiter. If not found, behavior falls back to rule (3).
        end_marker (str): The ending delimiter. If not found, behavior falls back to rule (2).
        strip (bool): Whether to trim whitespace from the result. Defaults to True.

    Returns:
        str | None: The extracted substring (possibly stripped), or None if no markers are found.
    """

    if not start_marker and not end_marker:
        return None  # At least one marker is required

    # Case 1: Both start and end markers exist in correct order
    if start_marker:
        s = text.find(start_marker)
        if s != -1:
            s2 = s + len(start_marker)
            if end_marker:
                e = text.find(end_marker, s2)
                if e != -1:
                    out = text[s2:e]
                    return out.strip() if strip else out

            # No valid end marker → take from start_marker to the end
            out = text[s2:]
            return out.strip() if strip else out

    # Case 2: No start marker (or it's after end marker), but end marker exists
    if end_marker:
        e = text.find(end_marker)
        if e != -1:
            out = text[:e]
            return out.strip() if strip else out

    # Case 3: Neither start nor end marker found
    return None


def extract_search_intent_and_query(text: str):
    """
    Parse the unique <|begin_search_query|>...<|end_search_query|> block emitted by the LRM.
    Returns:
      intent (str): The intent string.
      query (str): The query string.
    """
    BLOCK_RE = re.compile(
        r"<\|begin_search_query\|>(.*?)<\|end_search_query\|>",
        re.DOTALL | re.IGNORECASE,
    )
    INTENT_RE = re.compile(r"^\s*intent\s*:\s*(.+?)\s*$", re.IGNORECASE)
    QUERY_RE = re.compile(r"^\s*query\s*:\s*(.+?)\s*$", re.IGNORECASE)

    m = BLOCK_RE.search(text)
    if not m:
        logger.error(f"No <|begin_search_query|>...<|end_search_query|> block found.")
        return None, None

    block = m.group(1)
    intent, query = "", ""

    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        if mi := INTENT_RE.match(line):
            intent = mi.group(1).strip()
        elif mq := QUERY_RE.match(line):
            query = mq.group(1).strip()

    # Fallback strategy: take the last non-empty line if query is missing explicitly
    if not query:
        fallback_lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if fallback_lines:
            query = fallback_lines[-1]

    if not query:
        logger.error("Missing query in search block.")
        return None, None

    return intent, query


def format_search_results(
    url2pagecontent: dict, content_field: str | None = None
) -> str:
    """Render each `PageContent` using only title, snippet, ranking_id, url, site_name, and (optionally) the chosen content field."""

    formatted_chunks: list[str] = []

    for _, page_content in url2pagecontent.items():
        # Prepare sanitized values without mutating the cached PageContent
        title = (page_content.title or "").replace("<b>", "").replace("</b>", "")
        snippet = (page_content.snippet or "").replace("<b>", "").replace("</b>", "")
        ranking_id = getattr(page_content, "ranking_id", None)
        url = getattr(page_content, "url", None)
        site_name = getattr(page_content, "site_name", None)
        content_value = (
            getattr(page_content, content_field, None) if content_field else None
        )

        payload = {
            "title": title or None,
            "snippet": snippet or None,
            "ranking_id": ranking_id,
            "url": url,
            "site_name": site_name,
        }

        if content_field:
            payload[content_field] = content_value

        # Keep keys with non-empty values to reduce noise
        payload = {k: v for k, v in payload.items() if v not in (None, "")}

        rank_display = ranking_id + 1 if isinstance(ranking_id, int) else "Unknown"
        formatted_chunks.append(
            f"***Web Page {rank_display}:***\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )

    return "\n".join(formatted_chunks)


def extract_markdown_content(text):
    """Extract content between ``` and ``` tags."""
    pattern = r"```\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    return text


@dataclass
class ResponseContext:
    """Reasoning context for the current sequence"""

    question: str
    response: str
    article: str
    article_outline: str
    total_tokens: int
    max_completion_tokens: int
    writer_client: LLMClient
    reasoning_client: LLMClient
    args: argparse.Namespace
    user_agent: str
    cookie_jar: DeepResearchCookieJar
    seq: dict
    web_cache: SQLiteCache
    bm25: BM25Okapi | None
    document_memory: list
    emit: Optional[Callable[[dict], None]] = None


async def handle_write_section_response(context: ResponseContext):
    """
    Handle the response of write section task, which includes section_name and section_goal,
    then let writer llm to generate section content, and update the article and article_outline
    """
    # context.response = normalize_instruction_following_response(
    #     context.response, BEGIN_WRITE_SECTION, END_WRITE_SECTION
    # )
    section_task = extract_between(
        context.response, BEGIN_WRITE_SECTION, END_WRITE_SECTION
    )
    logger.info(f"Writing a section:\n{section_task}\n\n")
    error_message = (
        f"The Generated section_task is blank, the whole response is {context.response}"
    )
    if section_task is not None:
        # Use "\n" strictly to seperate section_name and section_goal
        section_parts = section_task.strip().split("\n", 1)
        if len(section_parts) == 2:
            section_name, section_goal = section_parts
            logger.info(
                f"The section name: \n{section_name}\nThe section goal: \n{section_goal}\n"
            )

            # Emit planned write section task
            if context.emit:
                try:
                    context.emit(
                        {
                            "type": "write_section_plan",
                            "payload": {
                                "section_name": section_name.strip(),
                                "section_goal": section_goal.strip(),
                            },
                        }
                    )
                except Exception:
                    pass

            # Prepare relevant documents using BM25
            if not context.bm25 and context.document_memory:
                tokenized_docs = [
                    word_tokenize_multilingual(doc.lower(), detect_language_zh_en(doc))
                    for doc in context.document_memory
                ]
                # TODO Use BM25S or drop BM25; switch to vector retrieval
                context.bm25 = BM25Okapi(tokenized_docs)

            if context.bm25:
                query = f"{section_name} {section_goal}"
                tokenized_query = word_tokenize_multilingual(
                    query.lower(), detect_language_zh_en(query)
                )
                doc_scores = context.bm25.get_scores(tokenized_query)
                # Get top 3 relevant documents
                # TODO First: is top-k=3 appropriate? Second: are returned docs useful or too noisy? Filter by score?
                top_indices = np.argsort(doc_scores)[-3:][::-1]
                relevant_documents = ""
                for i, idx in enumerate(top_indices, 1):
                    relevant_documents += (
                        f"Document {i}:\n{context.document_memory[idx]}\n\n"
                    )
            else:
                relevant_documents = ""

            # Generate section content
            section_prompt = get_write_section_instruction(
                question=context.question,
                # TODO Also pass the response to the writer? Could this exceed the context window or introduce irrelevant info?
                previous_thoughts=context.seq["response"],
                relevant_documents=relevant_documents,
                section_name=section_name,
                section_goal=section_goal,
                # TODO Only the current article outline is provided to the writer, not the full article
                current_article=context.article_outline,
                language=context.args.language,
            )

            section_response = await generate_response(
                # Writer llm to write this section content
                client=context.writer_client,
                prompt=section_prompt,
                max_completion_tokens=context.max_completion_tokens,
            )
            section_content = section_response.response_text
            # Update article
            section_content = (
                section_content.replace("## Section Name: ", "## ")
                .split("### Conclusion")[0]
                .split("### 结论")[0]
                .strip("\n")
                .strip()
            )
            section_content = re.sub(r"## Section \d+:", "##", section_content)
            # TODO Does section_name appear in section_content? If not, the section heading is missing and the article is incomplete
            context.article += f"\n{section_content}\n\n"

            # Extract outline by finding all headers
            headers = re.findall(r"^#{1,4}\s+.*$", context.article, re.MULTILINE)
            # Reconstruct the article outline
            context.article_outline = "\n".join(headers) + "\n"

            logger.info(f"Current Article: \n{context.article}\n")
            logger.info(f"Current Article Outline: \n{context.article_outline}\n")
            if context.emit:
                try:
                    context.emit(
                        {
                            "type": "outline_update",
                            "payload": {
                                "article_outline": context.article_outline,
                                "section_content": section_content,
                            },
                        }
                    )
                except Exception:
                    pass
        else:
            logger.info(error_message)
            return
    else:
        logger.info(error_message)
        return


async def handle_edit_article_response(context: ResponseContext):
    """
    Handle the response of edit article task, which edits the whole article
    """
    # context.response = normalize_instruction_following_response(
    #     context.response, BEGIN_EDIT_ARTICLE, END_EDIT_ARTICLE
    # )
    edit_instruction = extract_between(
        context.response, BEGIN_EDIT_ARTICLE, END_EDIT_ARTICLE
    )
    if edit_instruction is None or len(edit_instruction) <= 15:
        message = f"The Generated edit_instruction is blank, so nothing to edit, the whole response is {context.response}"
        logger.info(message)
        return

    logger.info(f"Generated a article edit task: \n{edit_instruction}")
    if context.emit:
        try:
            context.emit(
                {
                    "type": "edit_article",
                    "payload": {"instruction": edit_instruction},
                }
            )
        except Exception:
            pass
    if edit_instruction and context.article:
        edit_prompt = get_edit_article_instruction(
            edit_instruction, context.article, language=context.args.language
        )
        edit_response = await generate_response(
            client=context.writer_client,
            prompt=edit_prompt,
            max_completion_tokens=context.max_completion_tokens,
        )
        context.article = extract_markdown_content(edit_response.response_text)
        logger.info(f"Current Article Content: \n{context.article}\n")

    # Extract outline by finding all headers
    headers = re.findall(r"^#{1,4}\s+.*$", context.article, re.MULTILINE)
    # Reconstruct the article outline
    context.article_outline = "\n".join(headers) + "\n"
    if context.emit:
        try:
            context.emit(
                {
                    "type": "outline_update",
                    "payload": {"article_outline": context.article_outline},
                }
            )
        except Exception:
            pass


async def handle_check_article_response(context: ResponseContext):
    """
    Handle the response of check article outline task.
    In short, append the current article outline to the reasoning trajectory.
    """
    logger.info("Check article outline")

    # Check and add title if needed
    if not context.article.strip().startswith("# "):
        title_prompt = get_title_instruction(
            context.question, context.article, language=context.args.language
        )
        title_response = await generate_response(
            client=context.writer_client,
            prompt=title_prompt,
            max_completion_tokens=context.max_completion_tokens,
        )
        title = title_response.response_text
        title = title.replace("\n", "").strip('"').strip("'").strip()
        context.article = f"# {title}\n\n{context.article}"
        context.article_outline = f"# {title}\n\n{context.article_outline}"

    append_text = f"{BEGIN_CHECK_RESULT}{context.article_outline}{END_CHECK_RESULT}"
    context.seq["reasoning_trajectory"] += append_text
    context.seq["response"] += append_text  # TODO Append to response as well?
    context.seq["history"].append(append_text)  # TODO Should history be modified?
    if context.args.language == "en":
        context.total_tokens += len(append_text.split())
    elif context.args.language == "zh":
        context.total_tokens += len(append_text)
    logger.info(f"Current article outline: \n{context.article_outline}\n")
    if context.emit:
        try:
            context.emit(
                {
                    "type": "outline_update",
                    "payload": {"article_outline": context.article_outline},
                }
            )
        except Exception:
            pass


async def handle_search_query_response(context: ResponseContext):
    """
    Handle the response of search query subtask
    TODO Core logic; needs careful analysis and potential improvements
    """
    # context.response = normalize_instruction_following_response(
    #     context.response, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY
    # )
    # search_query = extract_between(
    #     context.response, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY
    # )
    search_intent, search_query = extract_search_intent_and_query(context.response)
    logger.info(f"Search intent: {search_intent}, search query: {search_query}")
    if context.emit and search_query:
        try:
            context.emit(
                {
                    "type": "search_query",
                    "payload": {"intent": search_intent, "query": search_query},
                }
            )
        except Exception:
            pass

    error_message = f"{search_query} is not valid, skip this search query subtask"
    if search_query is None or len(search_query) <= 5:
        logger.info(error_message)
        # Nothing to search, so skip this search query subtask
        return

    if search_query in INVALID_SEARCH_QUERY:
        logger.info(error_message)
        # Nothing to search, so skip this search query subtask
        return

    if search_query in context.seq["executed_search_queries"]:
        logger.info(
            f"{search_query} already executed; reuse previously gathered results"
        )
        # If search query was already executed, let lrm know to reuse previously gathered results
        append_text = f"\n\n{BEGIN_SEARCH_RESULT}You have already searched for this query. {END_SEARCH_RESULT}\n\nOK, let me use the previously found information."
        context.seq["reasoning_trajectory"] += append_text
        context.seq["response"] += append_text
        context.seq["history"].append(append_text)
        context.seq["search_count"] += 1
        if context.args.language == "en":
            context.total_tokens += len(append_text.split())
        elif context.args.language == "zh":
            context.total_tokens += len(append_text)
        # DO NOT SEARCH AGAIN
        return

    cached_urls = context.web_cache.get_search_urls(search_query)
    if cached_urls is not None:
        search_result_urls = cached_urls
    else:
        try:
            if context.args.search_engine == "cookie_google":
                search_result_urls = cookie_google_search(
                    search_query, context.user_agent, context.cookie_jar
                )
            elif context.args.search_engine == "duckduckgo":
                search_result_urls = duckduckgo_search(search_query, context.user_agent)
            else:
                error_message = f"Invalid search engine: {context.args.search_engine}, not supported yet."
                logger.error(error_message)
                raise ValueError(error_message)
            if len(search_result_urls) > 0:
                context.web_cache.set_search_urls(search_query, search_result_urls)
        except Exception as e:
            logger.info(
                f"Error during search '{search_query}' using {context.args.search_engine}: {e}"
            )
            # First fallback to DuckDuckGo (no cookie), then Serper if available
            try:
                search_result_urls = duckduckgo_search(search_query, context.user_agent)
                if len(search_result_urls) > 0:
                    context.web_cache.set_search_urls(search_query, search_result_urls)
                    logger.info(
                        f"Used DuckDuckGo fallback for '{search_query}', URLs: {search_result_urls}"
                    )
                else:
                    raise RuntimeError("DuckDuckGo returned no results")
            except Exception as e_ddg:
                logger.info(f"DuckDuckGo fallback failed: {e_ddg}")
                # Fallback to Serper if available
                api_key = os.getenv("SERPER_API_KEY")
                if api_key:
                    try:
                        serper_json = google_serper_search(search_query, api_key)
                        # Prefer organic results
                        organic = (
                            serper_json.get("organic", [])
                            if isinstance(serper_json, dict)
                            else []
                        )
                        search_result_urls = [
                            item.get("link") for item in organic if item.get("link")
                        ][:10]
                        search_result_urls = [u for u in search_result_urls if u]
                        logger.info(
                            f"Used Serper fallback for '{search_query}', URLs: {search_result_urls}"
                        )
                        if len(search_result_urls) > 0:
                            context.web_cache.set_search_urls(
                                search_query, search_result_urls
                            )
                    except Exception as e2:
                        logger.info(f"Serper fallback failed: {e2}")
                        search_result_urls = []
                else:
                    search_result_urls = []

    logger.info(
        f"Searched for '{search_query}' with {context.args.search_engine}; URLs: {search_result_urls}"
    )

    # Fetch page content for URLs that are not yet cached
    uncached_urls = []
    for url in search_result_urls:
        if not context.web_cache.has_url(url):
            uncached_urls.append(url)

    if uncached_urls:
        try:
            url2pagecontent = await fetch_page_content_async(
                uncached_urls,
                extractor=context.args.extractor,
                extractor_kwargs={
                    "user_agent": context.user_agent,
                    "llm_name": context.args.summarizer_model_name,
                    "llm_base_url": context.args.summarizer_model_base_url,
                    "llm_api_key": context.args.summarizer_model_api_key,
                    "question": context.question,
                    "search_query": search_query,
                },
                keep_links=context.args.keep_links,
            )
            for url, page_content in url2pagecontent.items():
                # page_content type is `PageContent`
                content = page_content.raw_content
                # Only cache content if it doesn't contain error indicators
                # TODO how to deal with error indicators and update the rules
                has_error = (
                    content is None
                    or (
                        any(
                            indicator.lower() in content.lower()
                            for indicator in ERROR_INDICATORS
                        )
                        and len(content.split()) < 5
                    )
                    or len(content) < 50
                    or len(content.split()) < 5
                )
                if not has_error:
                    context.web_cache.set_url_content(url, page_content)
        except Exception as e:
            logger.error(f"[Error] fetching URLs {uncached_urls}: {e}")

    # Get web page information for each result
    read_web_page = False  # TODO Why ALWAYS False
    url2pagecontent = {}
    for idx, url in enumerate(search_result_urls):
        page_content = context.web_cache.get_url_content(url)
        if page_content is None:
            continue
        else:
            # Failed to fetch the web page
            if page_content.raw_content is None or page_content.raw_content == "":
                continue

            # Set ranking id, starts from 0
            page_content.ranking_id = idx

            if idx < 5:
                if read_web_page:
                    context_chars = 10000
                else:
                    context_chars = 4000
            else:
                context_chars = 2000
            # Extract snippet context from web page context based on snippet
            _, snippet_context = extract_snippet_with_context(
                page_content.raw_content,
                page_content.snippet,
                context_chars=context_chars,
            )

        # Check if content has error indicators
        has_error = (
            any(
                indicator.lower() in snippet_context.lower()
                for indicator in ERROR_INDICATORS
            )
            or snippet_context == ""
        )
        if has_error:
            page_content.page_info = page_content.snippet
        else:
            if idx < 5 and read_web_page:  # TODO why 5?
                # Use detailed web page reader to process content
                reader_prompt = get_detailed_web_page_reader_instruction(
                    search_query, search_intent, snippet_context
                )
                candidate_info = await generate_response(
                    client=context.writer_client,
                    prompt=reader_prompt,
                    max_completion_tokens=8000,
                )
                candidate_info = candidate_info.response_text
                # page_content.page_info = page_info
            else:
                candidate_info = snippet_context

            # Skip manually concatenating `snippet`; the formatter(format_search_results) will includes it alongside `page_info`.
            # snippet_summary = (page_content.snippet or "").strip()
            # if snippet_summary:
            #     candidate_info = f"{snippet_summary}\n\n{candidate_info}".strip()

            page_content.page_info = candidate_info

        url2pagecontent[url] = page_content

    formatted_documents = format_search_results(
        url2pagecontent, content_field="page_info"
    )
    logger.info(f"Formatted search result documents: {formatted_documents}")
    # Emit simplified search results for UI
    if context.emit:
        try:
            results_payload = []
            for _, page_content in url2pagecontent.items():
                results_payload.append(
                    {
                        "title": (page_content.title or "")
                        .replace("<b>", "")
                        .replace("</b>", ""),
                        "snippet": (page_content.snippet or "")
                        .replace("<b>", "")
                        .replace("</b>", ""),
                        "url": getattr(page_content, "url", None),
                        "site_name": getattr(page_content, "site_name", None),
                        "ranking_id": getattr(page_content, "ranking_id", None),
                        "page_info": getattr(page_content, "page_info", None),
                    }
                )
            context.emit(
                {"type": "search_result", "payload": {"results": results_payload}}
            )
        except Exception:
            pass

    if context.args.is_using_deep_web_explore:
        # Run deep web explorer to go beyond the initial top-N SERP (Search Engine Results Page); those hits may still miss
        # critical details, so guided browsing/clicks help surface more relevant evidence.
        # TODO analysis is better than formatted_documents?????
        analysis, explorer_prompt = await generate_deep_web_explorer(
            reasoning_client=context.reasoning_client,
            writer_client=context.writer_client,
            question=context.question,
            search_query=search_query,
            search_intent=search_intent,
            document=formatted_documents,
            args=context.args,
            web_cache=context.web_cache,
            user_agent=context.user_agent,
            cookie_jar=context.cookie_jar,
        )
        extracted_info = analysis
        # Store web explorer input/output with all interactions
        context.seq["web_explorer"].append(
            {
                "search_query": search_query,
                "Input": explorer_prompt,
                "Output": analysis,
                "Extracted_info": extracted_info,
            }
        )
        logger.info(f"Returned search results:\n{extracted_info}\n")
    else:
        extracted_info = formatted_documents

    # Update sequence with search results
    append_text = f"\n\n{BEGIN_SEARCH_RESULT}{extracted_info}{END_SEARCH_RESULT}\n\n"
    context.seq["reasoning_trajectory"] += append_text
    context.seq["response"] += append_text
    context.seq["history"].append(append_text)
    context.seq["search_count"] += 1
    context.seq["executed_search_queries"].add(search_query)
    if context.args.language == "zh":
        context.total_tokens += len(append_text)
    elif context.args.language == "en":
        context.total_tokens += len(append_text.split())

    # Add retrieved content to document memory and rebuild BM25 index if needed
    document_memory_updated = False
    for url, page_content in url2pagecontent.items():
        page_info = page_content.page_info
        if page_info and page_info != "Can not fetch the page content.":
            context.document_memory.append(page_info)
            document_memory_updated = True
    # Rebuild BM25 index if document memory was updated
    if document_memory_updated:
        tokenized_docs = [
            word_tokenize_multilingual(doc.lower(), detect_language_zh_en(doc))
            for doc in context.document_memory
        ]
        # If document memory was updated, rebuild BM25 index
        context.bm25 = BM25Okapi(tokenized_docs)


async def generate_response(  # TODO Is async necessary?
    client: LLMClient,
    prompt: str,
    temperature: float | None = None,
    top_p: float | None = None,
    max_completion_tokens: int | None = None,
    repetition_penalty: float | None = None,
    top_k: int | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    stop: list[str] = [END_SEARCH_QUERY],
    timeout: int | None = None,
) -> LLMResponse:
    """
    Generate a single response with retry logic

    Args:
        client: The LLMClient for the model
        prompt: The prompt to generate a response for
        temperature: The temperature to use for the response
        top_p: The top_p to use for the response
        max_completion_tokens: The maximum number of tokens to generate
        repetition_penalty: The repetition penalty to use for the response
        top_k: The top_k to use for the response
        frequency_penalty: The frequency penalty to use for the response
        presence_penalty: The presence penalty to use for the response
        stop: The stop sequence to use for the response
        timeout: The timeout for the response, if not set will use the default timeout of the model

    Returns:
        response: The generated response
    """
    # cooperative cancel before expensive call
    await _check_cancelled()
    response = await client.chat_completion(
        prompt,
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=max_completion_tokens,
        repetition_penalty=repetition_penalty,
        top_k=top_k,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        timeout=timeout,
    )
    return response


async def generate_deep_web_explorer(
    reasoning_client: LLMClient,
    writer_client: LLMClient,
    question: str,
    search_query: str,
    search_intent: str,
    document: str,
    args: argparse.Namespace,
    web_cache: SQLiteCache,
    user_agent: str | None = None,
    cookie_jar: DeepResearchCookieJar | None = None,
) -> tuple[str, str]:
    """
    Generate deep web exploration with multiple search and click operations

    Args:
        reasoning_client: The LLMClient for reasoning llm model
        writer_client: The LLMClient for writer llm model
        question: The question to answer
        search_query: The search query to execute
        document: The web page content corresponding url
        search_intent: The search intent to use
        args: The arguments namespace
        web_cache: The web cache dictionary
        user_agent: The user agent to use

    Returns:
        A tuple of the output and the original prompt
    """
    message = f"Starting deep web explorer with query='{search_query}' and intent='{search_intent}'"
    logger.info(message)
    prompt = get_deep_web_explorer_instruction(
        original_question=question,
        search_query=search_query,
        search_intent=search_intent,
        search_result=document,
    )
    original_prompt = prompt
    output = ""
    total_tokens = len(prompt.split())  # English prompt template
    clicked_urls = set()  # Track clicked URLs
    executed_search_queries = (
        set()
    )  # Track executed search queries for deep web exploration
    total_interactions = 0
    finished = False
    latest_search_query = search_query

    while True:
        await _check_cancelled()
        model_response = await generate_response(
            client=reasoning_client,
            prompt=prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            max_completion_tokens=args.max_completion_tokens,
            repetition_penalty=args.repetition_penalty,
            top_k=args.top_k,
            frequency_penalty=args.frequency_penalty,
            presence_penalty=args.presence_penalty,
            stop=[END_SEARCH_QUERY, END_CLICK_LINK],
        )

        output += model_response.response_text.strip()
        _, response_tokens = extract_token_usage(model_response)
        if response_tokens > 0:
            total_tokens += response_tokens
        else:
            response_tokens = (
                len(model_response.response_text.split())
                if args.language == "en"
                else len(model_response.response_text)
            )
            total_tokens += response_tokens

        if (
            total_tokens >= DEEP_WEB_EXPLORE_MAX_TOKENS
            or total_interactions >= DEEP_WEB_EXPLORE_MAX_INTERACTIONS
        ):
            break

        if model_response.response_text.rstrip().endswith(
            END_SEARCH_QUERY
        ) or model_response.response_text.lstrip().startswith(BEGIN_SEARCH_QUERY):
            new_search_intent, new_search_query = extract_search_intent_and_query(
                model_response.response_text
            )
            total_interactions += 1
            if (
                new_search_query is None
                or len(new_search_query) <= 5
                or new_search_query in INVALID_SEARCH_QUERY
            ):
                error_message = f"The Generated search_query {new_search_query} is too short to search, \
                        the whole response is {model_response.response_text}"
                logger.error(error_message)
                continue

            if new_search_query in executed_search_queries:
                search_result = f"\n{BEGIN_SEARCH_RESULT}\nYou have already searched for this query. Please use the previously found information.\n{END_SEARCH_RESULT}\n"
                output += search_result
                prompt += output
                total_tokens += len(search_result.split())
                continue

            executed_search_queries.add(new_search_query)
            latest_search_query = new_search_query

            cached_urls = web_cache.get_search_urls(new_search_query)
            if cached_urls is not None:
                search_result_urls = cached_urls
            else:
                try:
                    if args.search_engine == "cookie_google":
                        search_result_urls = cookie_google_search(
                            new_search_query, user_agent, cookie_jar
                        )
                    elif args.search_engine == "duckduckgo":
                        search_result_urls = duckduckgo_search(
                            new_search_query, user_agent
                        )
                    else:
                        logger.error(f"Invalid search engine: {args.search_engine}")
                        search_result_urls = []
                    if len(search_result_urls) > 0:
                        web_cache.set_search_urls(new_search_query, search_result_urls)
                except Exception as e:
                    logger.info(f"Error during search query '{new_search_query}': {e}")
                    # First fallback to DuckDuckGo
                    try:
                        search_result_urls = duckduckgo_search(
                            new_search_query, user_agent
                        )
                        if len(search_result_urls) > 0:
                            web_cache.set_search_urls(
                                new_search_query, search_result_urls
                            )
                            logger.info(
                                f"Used DuckDuckGo fallback for '{new_search_query}', URLs: {search_result_urls}"
                            )
                        else:
                            raise RuntimeError("DuckDuckGo returned no results")
                    except Exception as e_ddg:
                        logger.info(f"DuckDuckGo fallback failed: {e_ddg}")
                        # Fallback to Serper if available
                        api_key = os.getenv("SERPER_API_KEY")
                        if api_key:
                            try:
                                serper_json = google_serper_search(
                                    new_search_query, api_key
                                )
                                organic = (
                                    serper_json.get("organic", [])
                                    if isinstance(serper_json, dict)
                                    else []
                                )
                                search_result_urls = [
                                    item.get("link")
                                    for item in organic
                                    if item.get("link")
                                ][:10]
                                search_result_urls = [
                                    u for u in search_result_urls if u
                                ]
                                if len(search_result_urls) > 0:
                                    web_cache.set_search_urls(
                                        new_search_query, search_result_urls
                                    )
                                logger.info(
                                    f"Used Serper fallback for '{new_search_query}', URLs: {search_result_urls}"
                                )
                            except Exception as e2:
                                logger.info(f"Serper fallback failed: {e2}")
                                search_result_urls = []
                        else:
                            search_result_urls = []

            logger.info(
                f'Searched for "{new_search_query}"; URLs: {search_result_urls}'
            )

            uncached_urls = [
                url for url in search_result_urls if not web_cache.has_url(url)
            ]
            if uncached_urls:
                try:
                    await _check_cancelled()
                    url2pagecontent = await fetch_page_content_async(
                        uncached_urls,
                        extractor=args.extractor,
                        extractor_kwargs={
                            "user_agent": user_agent,
                            "llm_name": args.summarizer_model_name,
                            "llm_base_url": args.summarizer_model_base_url,
                            "llm_api_key": args.summarizer_model_api_key,
                            "question": question,
                            "search_query": new_search_query,
                        },
                        keep_links=args.keep_links,
                    )
                    for url, page_content in url2pagecontent.items():
                        # page_content type is `PageContent`
                        content = page_content.raw_content
                        has_error = (
                            content is None
                            or (
                                any(
                                    indicator.lower() in content.lower()
                                    for indicator in ERROR_INDICATORS
                                )
                                and len(content.split()) < 5
                            )
                            or len(content) < 50
                            or len(content.split()) < 5
                        )
                        if not has_error:
                            web_cache.set_url_content(url, page_content)
                except Exception as e:
                    logger.error(f"[Error] fetching URLs {uncached_urls}: {e}")

            url2pagecontent = {}
            for idx, url in enumerate(search_result_urls):
                page_content = web_cache.get_url_content(url)
                if page_content is None:
                    continue
                if page_content.raw_content is None or page_content.raw_content == "":
                    continue
                page_content.ranking_id = idx
                url2pagecontent[url] = page_content

            formatted_documents = format_search_results(
                url2pagecontent, content_field=None
            )

            # Append search results
            search_result = (
                f"\n{BEGIN_SEARCH_RESULT}\n{formatted_documents}\n{END_SEARCH_RESULT}\n"
            )
            output += search_result
            prompt += output
            if args.language == "zh":
                total_tokens += len(search_result)
            elif args.language == "en":
                total_tokens += len(search_result.split())

        # Check for click link
        elif model_response.response_text.rstrip().endswith(
            END_CLICK_LINK
        ) or model_response.response_text.lstrip().startswith(BEGIN_CLICK_LINK):
            url = extract_between(
                model_response.response_text, BEGIN_CLICK_LINK, END_CLICK_LINK
            )
            total_interactions += 1
            if url is None or len(url) <= 5:
                continue
            if url and (url in clicked_urls):
                # If URL was already clicked, append message
                click_result = f"\n{BEGIN_CLICK_RESULT}\nYou have already clicked this URL.\n{END_CLICK_RESULT}\nOK, let me use the previously found information."
                output += click_result
                prompt += output
                total_tokens += len(click_result.split())
                continue

            click_intent_response = await generate_response(
                client=writer_client,
                prompt=get_click_intent_instruction(question, output),
                max_completion_tokens=args.max_completion_tokens,
            )

            if url and click_intent_response.response_text:
                clicked_urls.add(url)  # Add URL to clicked set
                logger.info(
                    f"Clicking on URL: {url} with intent: {click_intent_response.response_text}"
                )
                # Fetch and process page content
                if not web_cache.has_url(url):
                    try:
                        url2pagecontent = await fetch_page_content_async(
                            [url],
                            extractor=args.extractor,
                            extractor_kwargs={
                                "user_agent": user_agent,
                                "llm_name": args.summarizer_model_name,
                                "llm_base_url": args.summarizer_model_base_url,
                                "llm_api_key": args.summarizer_model_api_key,
                                "question": question,
                                "search_query": latest_search_query,  # TODO Any logic flaw with latest_search_query here? Must a click always have a search query? Can the LLM summarizer run without it?
                            },
                            keep_links=args.keep_links,
                        )
                        page_text = url2pagecontent[url].raw_content
                        # Only cache content if it doesn't contain error indicators
                        has_error = (
                            page_text is None
                            or any(
                                indicator.lower() in page_text.lower()
                                for indicator in ERROR_INDICATORS
                            )
                            and len(page_text.split()) < 5
                        ) or page_text == ""
                        if not has_error:
                            web_cache.set_url_content(url, url2pagecontent[url])
                    except Exception as e:
                        logger.error(f"Error fetching URL {url}: {e}")
                        page_text = ""
                else:
                    cached_page = web_cache.get_url_content(url)
                    page_text = cached_page.raw_content if cached_page else ""

                # Check if content has error indicators
                has_error = (
                    page_text is None
                    or any(
                        indicator.lower() in page_text.lower()
                        for indicator in ERROR_INDICATORS
                    )
                    or page_text == ""
                )

                if has_error:
                    # If page_text has error, use it directly as summary
                    summary = "Unable to fetch the page text. You can try other links."
                else:
                    # Use web page reader to summarize page_text
                    reader_prompt = get_click_web_page_reader_instruction(
                        click_intent_response.response_text, page_text[:20000]
                    )
                    await _check_cancelled()
                    summary_response = await generate_response(
                        client=writer_client,
                        prompt=reader_prompt,
                        max_completion_tokens=8000,
                    )
                    summary = summary_response.response_text

                # Append click results
                click_result = (
                    f"\n{BEGIN_CLICK_RESULT}\n{summary}\n{END_CLICK_RESULT}\n"
                )
                output += click_result
                prompt += output
                total_tokens += len(click_result.split())  # TODO use usage token_counts

        else:
            finished = True
            break

    # Add max limit message if needed
    if not finished and (
        total_tokens >= DEEP_WEB_EXPLORE_MAX_TOKENS
        or total_interactions >= DEEP_WEB_EXPLORE_MAX_INTERACTIONS
    ):
        output += f"\n{BEGIN_CLICK_RESULT}\nYou have reached the limit for clicking links.\n{END_CLICK_RESULT}\n\nOK, I will now provide the final information based on my collected information.\n\n**Final Information:**"
        prompt += output
        final_response = await generate_response(
            client=reasoning_client,
            prompt=prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            max_completion_tokens=512,
            repetition_penalty=1.2,
            top_k=args.top_k,
            frequency_penalty=args.frequency_penalty,
            presence_penalty=args.presence_penalty,
        )
        output += final_response.response_text

    return output, original_prompt


async def process_single_sequence(
    seq: dict,
    reasoning_client: LLMClient,
    writer_client: LLMClient,
    args: argparse.Namespace,
    web_cache: SQLiteCache,
    user_agent: str,
    cookie_jar: DeepResearchCookieJar,
    emit: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Process a single sequence through its entire reasoning chain with REASONING_TRAJECTORY_MAX_TOKENS limit"""

    # Early cancel check
    await _check_cancelled(emit)

    # Initialize limits
    total_interactions = 0  # Track total interactions

    # Generate search plan first
    logger.info(
        f"Generating search plan via {args.reasoning_model_name} for {seq['item']['Question']} ..."
    )
    question = seq["item"]["Question"]
    # TODO The search plan is static and non-adaptive; it's just part of the reasoning trajectory, and the LRM may not follow it
    search_plan_prompt = get_search_plan_instruction(
        question,
        max_tool_calls=args.max_tool_calls,
        max_plan_steps=args.max_plan_steps,
        language=args.language,
    )
    # Inform UI that search plan generation starts, and start a gentle ticker
    if emit:
        try:
            emit(
                {
                    "type": "status",
                    "payload": {
                        "stage": "search_plan",
                        "message": "Generating search plan...",
                    },
                }
            )
        except Exception:
            pass

    async def _ticker():
        try:
            elapsed = 0
            while True:
                await asyncio.sleep(2)
                elapsed += 2
                if emit:
                    try:
                        emit(
                            {
                                "type": "status",
                                "payload": {
                                    "stage": "search_plan",
                                    "message": "Still thinking...",
                                    "elapsed": elapsed,
                                },
                            }
                        )
                    except Exception:
                        pass
        except asyncio.CancelledError:
            return

    ticker_task = asyncio.create_task(_ticker())
    await _check_cancelled(emit)

    search_plan_response = await generate_response(
        # Using reasong llm to generate search plan
        client=reasoning_client,
        prompt=search_plan_prompt,
        max_completion_tokens=args.max_completion_tokens,
        # Use custom search plan timeout, which is longer than the default timeout
        timeout=args.search_plan_timeout,
    )
    # Stop ticker and notify
    try:
        ticker_task.cancel()
    except Exception:
        pass
    if emit:
        try:
            emit(
                {
                    "type": "status",
                    "payload": {
                        "stage": "search_plan",
                        "message": "Search plan received.",
                    },
                }
            )
        except Exception:
            pass

    logger.info(f"Generated Search plan:\n{search_plan_response.response_text}\n\n")
    if search_plan_response.response_text == "":
        error_message = """No search plan generated. Maybe you can 
            increase timeout to wait for the model to generate a search plan."""
        logger.error(error_message)
        raise ValueError(error_message)
    # Emit search plan event
    if emit:
        try:
            emit(
                {
                    "type": "search_plan",
                    "payload": {
                        "plan": search_plan_response.response_text,
                        "prompt": search_plan_prompt[:2000],
                    },
                }
            )
        except Exception:
            pass

    # Generate the full instruction with the plan
    user_prompt = get_report_webthinker_instruction(
        question, search_plan_response.response_text, language=args.language
    )
    seq["reasoning_trajectory"] = user_prompt
    seq["original_prompt"] = user_prompt  # Just for logging

    # Initialize web explorer interactions list and article-related variables
    seq["web_explorer"] = []
    article = ""  # Generated report
    article_outline = ""  # Outline of the article
    document_memory = []  # Store **ALL** retrieved web page content

    # Initialize BM25 for document retrieval
    tokenized_docs = []
    bm25 = None

    # First response uses chat completion
    model_response = await generate_response(
        client=reasoning_client,
        prompt=seq["reasoning_trajectory"],
        temperature=args.temperature,
        top_p=args.top_p,
        max_completion_tokens=args.max_completion_tokens,
        repetition_penalty=args.repetition_penalty,
        top_k=args.top_k,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
        stop=[  # No <|end_think|>
            END_SEARCH_QUERY,
            END_WRITE_SECTION,
            END_EDIT_ARTICLE,
            BEGIN_CHECK_ARTICLE,
        ],
    )
    logger.info(
        f"First response from {args.reasoning_model_name}:\n{model_response.response_text}"
    )
    prompt_tokens, response_tokens = extract_token_usage(model_response)
    if prompt_tokens > 0:
        total_tokens = prompt_tokens + response_tokens
    else:
        prompt_tokens = (
            len(seq["reasoning_trajectory"].split())
            if args.language == "en"
            else len(seq["reasoning_trajectory"])
        )
        response_tokens = (
            len(model_response.response_text.split())
            if args.language == "en"
            else len(model_response.response_text)
        )

        total_tokens = prompt_tokens + response_tokens

    seq["response"] += model_response.response_text
    seq["history"].append(model_response.response_text)
    # TODO Should there be a separator between prompt and response?
    seq["reasoning_trajectory"] += model_response.response_text
    if emit:
        try:
            emit(
                {"type": "reasoning", "payload": {"text": model_response.response_text}}
            )
        except Exception:
            pass

    while not seq["finished"]:
        # Check interaction limit
        if total_interactions >= MAX_INTERACTIONS:
            message = f"Reached maximum interaction limit {MAX_INTERACTIONS}, deep research will be finished"
            logger.info(message)
            seq["finished"] = True
            break

        remaining_token_budget = max(REASONING_TRAJECTORY_MAX_TOKENS - total_tokens, 0)
        if remaining_token_budget == 0:
            message = "Reached maximum token limit before handling the response, deep research will be finished"
            logger.info(message)
            seq["finished"] = True
            break

        # Normalize tags
        model_response.response_text = normalize_tagged_text(
            model_response.response_text
        )

        response_context = ResponseContext(
            question=question,
            response=model_response.response_text,
            article=article,
            article_outline=article_outline,
            total_tokens=total_tokens,
            max_completion_tokens=args.max_completion_tokens,
            writer_client=writer_client,
            reasoning_client=reasoning_client,
            args=args,
            user_agent=user_agent,
            cookie_jar=cookie_jar,
            seq=seq,
            web_cache=web_cache,
            bm25=bm25,
            document_memory=document_memory,
            emit=emit,
        )

        ##### Handle different response endings #####
        # Generated a section writting task, which includes section_name and section_goal
        # Calling writer llm to write a section
        if model_response.response_text.rstrip().endswith(
            END_WRITE_SECTION
        ) or model_response.response_text.lstrip().startswith(BEGIN_WRITE_SECTION):
            total_interactions += 1
            # Calling writer llm, so need to set correct max_completion_tokens for writer llm
            await handle_write_section_response(response_context)
            article = response_context.article
            article_outline = response_context.article_outline

        # Generated a article editting task
        # Calling writer llm to edit article
        elif model_response.response_text.rstrip().endswith(
            END_EDIT_ARTICLE
        ) or model_response.response_text.lstrip().startswith(BEGIN_EDIT_ARTICLE):
            total_interactions += 1
            await handle_edit_article_response(response_context)
            article = response_context.article
            article_outline = response_context.article_outline

        # Generated a article checking task
        # Calling writer llm to write article title if needed then append article outline to reasoning trajectory.
        # TODO The LRM's reasoning trajectory lacks the article content, so it doesn't know the article's actual text—at most the outline
        # In the section-writing task, the LRM also doesn't see the section output. Should we add a 'check section' task or a broader 'article refine'? (Does 'edit article' satisfy this need? Without access to the article, how can it design edit instructions? This is problematic.)
        elif model_response.response_text.rstrip().endswith(
            BEGIN_CHECK_ARTICLE
        ) or model_response.response_text.lstrip().startswith(BEGIN_CHECK_ARTICLE):
            total_interactions += 1
            await handle_check_article_response(response_context)
            article = response_context.article
            article_outline = response_context.article_outline
            total_tokens = response_context.total_tokens

        # Generated a search engine call query
        # Calling writer llm
        # TODO Review SearchControlModule per ChatGPT chat logs
        elif model_response.response_text.rstrip().endswith(
            END_SEARCH_QUERY
        ) or model_response.response_text.lstrip().startswith(BEGIN_SEARCH_QUERY):
            total_interactions += 1
            await handle_search_query_response(response_context)
            article = response_context.article
            article_outline = response_context.article_outline
            total_tokens = response_context.total_tokens
            seq = response_context.seq
            web_cache = response_context.web_cache
            bm25 = response_context.bm25
            document_memory = response_context.document_memory

        elif model_response.response_text.rstrip().endswith(
            END_THINK
        ) or model_response.response_text.lstrip().startswith(BEGIN_THINK):
            total_interactions += 1

        # Response not return any special end of sequence token, so deep research will be finished
        else:
            message = "Response not return any special end of sequence token, so deep research will be finished."
            logger.info(message)
            seq["finished"] = True
            break

        # If reached maximum token limit, deep research will be finished
        if total_tokens >= REASONING_TRAJECTORY_MAX_TOKENS:
            logger.info("Reached maximum token limit, deep research will be finished")
            seq["finished"] = True
            break
        # Continue to generate response
        else:
            logger.info("Go on reasoning...")
            model_response = await generate_response(
                client=reasoning_client,
                prompt=seq["reasoning_trajectory"],
                temperature=args.temperature,
                top_p=args.top_p,
                max_completion_tokens=args.max_completion_tokens,
                repetition_penalty=args.repetition_penalty,
                top_k=args.top_k,
                frequency_penalty=args.frequency_penalty,
                presence_penalty=args.presence_penalty,
                stop=[
                    END_SEARCH_QUERY,
                    END_WRITE_SECTION,
                    END_EDIT_ARTICLE,
                    BEGIN_CHECK_ARTICLE,
                ],
            )
            logger.info(f"Response:\n {model_response.response_text}\n\n")

            # Update token count and sequence fields
            _, response_tokens = extract_token_usage(model_response)
            if response_tokens > 0:
                total_tokens += response_tokens
            else:
                response_tokens = (
                    len(model_response.response_text.split())
                    if args.language == "en"
                    else len(model_response.response_text)
                )
                total_tokens += response_tokens
            # TODO We've already appended the response into reasoning_trajectory; should we verify formatting? Avoid duplicate appends
            seq["response"] += model_response.response_text
            seq["history"].append(model_response.response_text)
            seq["reasoning_trajectory"] += model_response.response_text
            if emit:
                try:
                    emit(
                        {
                            "type": "reasoning",
                            "payload": {"text": model_response.response_text},
                        }
                    )
                except Exception:
                    pass

    # Add final refinement step for the article using writer llm
    if article.strip():  # Only refine if article is not empty
        logger.info("---Getting final article...---")
        final_report_prompt = get_final_report_instruction(
            question, article, language=args.language
        )
        final_report_response = await generate_response(
            client=writer_client,
            prompt=final_report_prompt,
            max_completion_tokens=args.max_completion_tokens,  # Use a larger token limit for the final report
        )
        refined_article = extract_markdown_content(final_report_response.response_text)
        if refined_article.strip():  # Ensure refined article is not empty
            article = refined_article
            logger.info(f"---Final Article:---\n{article}\n")
        else:
            logger.info("---Refinement resulted in empty article, keeping original.---")

    # Store final article in sequence
    seq["article"] = article
    seq["article_outline"] = (
        article_outline  # Note: article_outline is not refined here
    )
    return seq


async def run_sequence(
    question: str,
    args: Optional[Union[argparse.Namespace, Mapping[str, Any]]] = None,
    emit: Optional[Callable[[dict], None]] = None,
    await_user_reply: Optional[Callable[[dict], Any]] = None,
) -> dict:
    """Programmatic entrypoint to run a single sequence and optionally emit JSONL events via `emit`.

    - Builds args from defaults and env if not provided.
    - Emits events when `emit` is provided; logging to files continues regardless.
    - Returns the final sequence dict (same structure as CLI flow).
    """
    # Build args Namespace from defaults (by calling parse_args in a sandboxed argv)
    if args is None or isinstance(args, Mapping):
        # Snapshot argv and replace with only the program name to avoid parsing the JSON payload
        old_argv = sys.argv[:]
        try:
            sys.argv = [old_argv[0]]
            base_args = parse_args()
        finally:
            sys.argv = old_argv
        if isinstance(args, Mapping):
            for k, v in args.items():
                setattr(base_args, k, v)
        args_ns = base_args
    elif isinstance(args, argparse.Namespace):
        args_ns = args
    else:
        raise TypeError("args must be None, a mapping, or an argparse.Namespace")

    # Load api keys from .env file, if not set, these api keys will be None
    args_ns.reasoning_model_name = (
        args_ns.reasoning_model_name
        if args_ns.reasoning_model_name
        else os.getenv("REASONING_MODEL_NAME")
    )
    args_ns.reasoning_model_base_url = os.getenv("REASONING_MODEL_BASE_URL")
    args_ns.reasoning_model_api_key = os.getenv("REASONING_MODEL_API_KEY")

    args_ns.writer_model_name = (
        args_ns.writer_model_name
        if args_ns.writer_model_name
        else os.getenv("WRITER_MODEL_NAME")
    )
    args_ns.writer_model_base_url = os.getenv("WRITER_MODEL_BASE_URL")
    args_ns.writer_model_api_key = os.getenv("WRITER_MODEL_API_KEY")

    args_ns.summarizer_model_name = (
        args_ns.summarizer_model_name
        if args_ns.summarizer_model_name
        else os.getenv("SUMMARIZER_MODEL_NAME")
    )
    args_ns.summarizer_model_base_url = os.getenv("SUMMARIZER_MODEL_BASE_URL")
    args_ns.summarizer_model_api_key = os.getenv("SUMMARIZER_MODEL_API_KEY")

    # Setup logger
    _ = get_logger(args_ns.log_dir, args_ns.log_file)
    set_seed(args_ns.seed)

    if not question:
        raise ValueError("question is required")
    logger.info(f"User Question: {question}\n")
    args_ns.language = detect_language_zh_en(question)

    # ---------------------- Caching Mechanism ----------------------
    cache_dir = args_ns.cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    sqlite_cache_path = os.path.join(cache_dir, f"{args_ns.search_engine}_cache.sqlite")
    web_cache = SQLiteCache(sqlite_cache_path)

    # Initialize the Reasoning llm client
    reasoning_client = create_llm_client(
        args_ns.reasoning_model_name,
        args_ns.reasoning_model_api_key,
        args_ns.reasoning_model_base_url,
        args_ns.timeout,
        args_ns.max_retries,
        args_ns.retry_delay,
    )
    # Initialize the writer llm client
    writer_client = create_llm_client(
        args_ns.writer_model_name,
        args_ns.writer_model_api_key,
        args_ns.writer_model_base_url,
        args_ns.timeout,
        args_ns.max_retries,
        args_ns.retry_delay,
    )

    # Initialized UserAgent and Cookie
    cookie_jar = load_cookies((args_ns.browser_type,))
    user_agent = get_browser_user_agent(
        args_ns.browser_type, use_selenium=args_ns.use_selenium
    )

    try:
        sequence = {
            "item": {
                "Question": question,
            },
            "reasoning_trajectory": "",
            "response": "",
            "finished": False,
            "history": [],
            "search_count": 0,
            "executed_search_queries": set(),
        }

        start_time = time.time()

        # Query clarification (best-effort; interactive if desktop cooperates)
        try:
            query_clarification_prompt = get_query_clarification_instruction(question)
            logger.info(f"Query Clarification Prompt:\n{query_clarification_prompt}\n")
            clarification_response = await generate_response(
                client=reasoning_client,
                prompt=query_clarification_prompt,
                timeout=args_ns.search_plan_timeout,
            )
            clarification_content = clarification_response.response_text or ""
            logger.info(f"Query Clarification Response:\n{clarification_content}\n")
            if emit and clarification_content.strip():
                try:
                    emit(
                        {
                            "type": "clarification_request",
                            "payload": {"message": clarification_content},
                        }
                    )
                except Exception:
                    pass

            # Await user reply if a handler is provided
            user_clarification = None
            if await_user_reply:
                try:
                    maybe = await_user_reply(
                        {
                            "type": "clarification",
                            "payload": {"prompt": clarification_content},
                        }
                    )
                    if asyncio.iscoroutine(maybe):
                        maybe = await maybe
                    user_clarification = maybe
                except Exception:
                    user_clarification = None

            # If user provided additional details, append to question and record
            added_text = None
            if isinstance(user_clarification, dict):
                added_text = user_clarification.get("text") or user_clarification.get(
                    "answer"
                )
            elif isinstance(user_clarification, str):
                added_text = user_clarification

            if isinstance(added_text, str) and added_text.strip():
                logger.info(f"Additional Details: {added_text}\n")
                _original_question_before_rewrite = question
                question_rewriting_prompt = get_query_rewriting_instruction(
                    question, added_text
                )
                logger.info(f"Query Rewriting Prompt:\n{question_rewriting_prompt}\n")
                question_rewriting_response = await generate_response(
                    client=reasoning_client,
                    prompt=question_rewriting_prompt,
                    timeout=args_ns.search_plan_timeout,
                )
                question = question_rewriting_response.response_text or ""
                # TODO Maybe just append added_text to question
                # question = f"{question}\n\n[User Clarifications]\n{added_text.strip()}"
                logger.info(f"Rewritten question: {question}")
                # Notify UI about the rewritten/clarified question
                if emit:
                    try:
                        emit(
                            {
                                "type": "clarification_rewrite",
                                "payload": {
                                    "original": _original_question_before_rewrite,
                                    "added": added_text.strip(),
                                    "rewritten": question,
                                },
                            }
                        )
                    except Exception:
                        pass
                # Reflect in sequence item for downstream prompts
                sequence["item"]["Question"] = question
                sequence["clarification"] = {
                    "prompt": clarification_content,
                    "answer": added_text.strip(),
                }
        except Exception:
            # Clarification is optional; proceed on any failure
            pass

        # Process single sequence directly
        await process_single_sequence(
            seq=sequence,
            reasoning_client=reasoning_client,
            writer_client=writer_client,
            args=args_ns,
            web_cache=web_cache,
            user_agent=user_agent,
            cookie_jar=cookie_jar,
            emit=emit,
        )

        # Convert set to list for JSON-serializable structure
        if isinstance(sequence.get("executed_search_queries"), set):
            sequence["executed_search_queries"] = sorted(
                sequence["executed_search_queries"]
            )

        logger.info(f"Process completed in {time.time() - start_time:.2f} seconds.")
        return sequence
    except asyncio.CancelledError:
        if emit:
            try:
                emit({"type": "cancelled", "payload": {"message": "Cancelled by user"}})
            except Exception:
                pass
        return {
            "item": {"Question": question},
            "reasoning_trajectory": "",
            "response": "",
            "finished": False,
            "history": [],
            "search_count": 0,
            "executed_search_queries": [],
        }
    finally:
        web_cache.close()


async def main_async():
    # Initialize args
    args = parse_args()
    # Load api keys from .env file, if not set, these api keys will be None
    args.reasoning_model_name = (
        args.reasoning_model_name
        if args.reasoning_model_name
        else os.getenv("REASONING_MODEL_NAME")
    )
    args.reasoning_model_base_url = os.getenv("REASONING_MODEL_BASE_URL")
    args.reasoning_model_api_key = os.getenv("REASONING_MODEL_API_KEY")

    args.writer_model_name = (
        args.writer_model_name
        if args.writer_model_name
        else os.getenv("WRITER_MODEL_NAME")
    )
    args.writer_model_base_url = os.getenv("WRITER_MODEL_BASE_URL")
    args.writer_model_api_key = os.getenv("WRITER_MODEL_API_KEY")

    args.summarizer_model_name = (
        args.summarizer_model_name
        if args.summarizer_model_name
        else os.getenv("SUMMARIZER_MODEL_NAME")
    )
    args.summarizer_model_base_url = os.getenv("SUMMARIZER_MODEL_BASE_URL")
    args.summarizer_model_api_key = os.getenv("SUMMARIZER_MODEL_API_KEY")

    # Setup logger
    logger = get_logger(args.log_dir, args.log_file)
    logger.info("Hyperparameters: --------------------------------")
    for k, v in vars(args).items():
        logger.info(f"          {k}: {v}")
    logger.info("-------------------------------------------------")
    set_seed(args.seed)

    if args.single_question is None:
        logger.error(
            "Error: --single_question is not provided, please provide a question to generate report."
        )
        return

    args.language = detect_language_zh_en(args.single_question)

    # ---------------------- Caching Mechanism ----------------------
    cache_dir = args.cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    sqlite_cache_path = os.path.join(cache_dir, f"{args.search_engine}_cache.sqlite")
    web_cache = SQLiteCache(sqlite_cache_path)

    # Define output directory
    output_dir = f"./outputs/{args.reasoning_model_name.lower()}.{args.writer_model_name.lower()}.deepresearch"
    os.makedirs(output_dir, exist_ok=True)
    markdown_dir = os.path.join(output_dir, "markdown")
    os.makedirs(markdown_dir, exist_ok=True)

    # Initialize the Reasoning llm client
    reasoning_client = create_llm_client(
        args.reasoning_model_name,
        args.reasoning_model_api_key,
        args.reasoning_model_base_url,
        args.timeout,
        args.max_retries,
        args.retry_delay,
    )
    # Initialize the writer llm client
    writer_client = create_llm_client(
        args.writer_model_name,
        args.writer_model_api_key,
        args.writer_model_base_url,
        args.timeout,
        args.max_retries,
        args.retry_delay,
    )

    # Initialized UserAgent and Cookie
    cookie_jar = load_cookies((args.browser_type,))
    user_agent = get_browser_user_agent(
        args.browser_type, use_selenium=args.use_selenium
    )

    try:
        sequence = {
            "item": {
                "Question": args.single_question,
            },
            "reasoning_trajectory": "",  # Corresponds to TIR's reasoning trajectory
            "response": "",
            "finished": False,
            "history": [],
            "search_count": 0,
            "executed_search_queries": set(),
        }

        # Log running time
        start_time = time.time()

        # Process single sequence directly
        await process_single_sequence(
            seq=sequence,
            reasoning_client=reasoning_client,
            writer_client=writer_client,
            args=args,
            web_cache=web_cache,
            user_agent=user_agent,
            cookie_jar=cookie_jar,
        )

        if sequence["article"].strip():
            markdown_filename = "article.md"
            # Add question as context at the top of the file
            question_context = f"Question: {sequence['item']['Question']}\n\n"

            with open(
                os.path.join(markdown_dir, markdown_filename), "w", encoding="utf-8"
            ) as f:
                f.write(question_context + sequence["article"])

            web_cache.save_report(
                question=args.single_question,
                article=sequence["article"],
                outline=sequence.get("article_outline", ""),
                metadata={
                    "reasoning_model": args.reasoning_model_name,
                    "writer_model": args.writer_model_name,
                    "summarizer_model": args.summarizer_model_name,
                    "language": args.language,
                },
            )

        result_json_name = "result.json"
        if isinstance(sequence.get("executed_search_queries"), set):
            # Convert to a JSON-serializable type before dumping the result artifact
            sequence["executed_search_queries"] = sorted(
                sequence["executed_search_queries"]
            )

        with open(
            os.path.join(output_dir, result_json_name), mode="w", encoding="utf-8"
        ) as json_file:
            json.dump(sequence, json_file, indent=4, ensure_ascii=False)

        logger.info(f"Process completed in {time.time() - start_time:.2f} seconds.")
    finally:
        web_cache.close()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
