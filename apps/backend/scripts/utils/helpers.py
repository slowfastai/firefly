import re
import json
import time
import sqlite3
from typing import Iterable, Any
import langid
from loguru import logger
from cutword import Cutter
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from utils.page_content import PageContent


def detect_language_zh_en(input_str: str):
    """
    Check if the input string is Chinese or English.
    # TODO support more languages

    Args:
        input_str (str): The input string to check.

    Returns:
        str: "zh" if the input string is Chinese, "en" otherwise.
    """
    assert isinstance(input_str, str), input_str
    if len(input_str) == 0:
        return "en"
    detect_result = langid.classify(input_str)
    if detect_result[0] == "zh":
        return "zh"
    else:
        return "en"


def word_tokenize_multilingual(text: str, language: str = "en") -> list[str]:
    """Word tokenize text with support for both English and Chinese."""
    if language == "zh":
        cutter = Cutter(want_long_word=True)
        return cutter.cutword(text)
    elif language == "en":
        return word_tokenize(text)
    else:
        raise ValueError(f"Unsupported language {language}")


def sent_tokenize_multilingual(text: str, language: str = "en") -> list[str]:
    """Sentence tokenize text with support for both English and Chinese."""
    if language == "zh":

        def sent_tokenize_zh_regex(text):
            # Use a regular expression to capture Chinese sentence-ending punctuation
            # TODO Spacy or other NLP library
            sentences = re.split(r"([。！？\?])", text)
            new_sents = []
            for i in range(int(len(sentences) / 2)):
                sent = sentences[2 * i] + sentences[2 * i + 1]
                new_sents.append(sent)
            if len(sentences) % 2 == 1 and sentences[-1].strip() != "":
                new_sents.append(sentences[-1])
            return new_sents

        return sent_tokenize_zh_regex(text)

    elif language == "en":
        return sent_tokenize(text)
    else:
        raise ValueError(f"Unsupported language {language}")


def is_o_or_gpt5(model: str) -> bool:
    """
    Return True only if using OpenAI's official endpoint (or proxy endpoint) AND the model belongs to
    the OpenAI o-series (o1/o3/o4) or gpt-5 family.

    Purpose:
    - Used to select Chat Completions parameters specific to o/gpt-5 models
      (e.g., prefer `max_completion_tokens`, ignore `temperature`/`top_p`).

    Args:
        model: Full model name (e.g., "o3-2025-04-16", "gpt-5.1").

    Returns:
        True if both conditions hold: (a) model is o*/gpt-5, and (b) base_url is None;
        otherwise False.
    """
    n = model.lower()
    name_condiation = (
        n.startswith(("o1", "o3", "o4")) or n.startswith("gpt-5") or ("gpt-5" in n)
    )
    # url_condition = base_url is None
    # return name_condiation and url_condition
    return name_condiation


class SQLiteCache:
    """Simple SQLite-backed cache for search results and URL content."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self._create_tables()

    def _create_tables(self) -> None:
        """Ensure the SQLite schema exists for both search and URL caches."""
        with self.conn:
            # search_cache stores one row per query along with the serialized URL list and a freshness timestamp:
            #   - query: text of the search query (primary key)
            #   - urls: JSON string of the URL results for that query
            #   - updated_at: Unix epoch recording when we last wrote this row
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS search_cache (
                    query TEXT PRIMARY KEY,
                    urls TEXT NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """
            )
            # url_cache indexes the full PageContent payload by URL, again with an updated_at column:
            #   - url: canonical URL acting as the primary key
            #   - payload: JSON-serialized PageContent structure
            #   - updated_at: Unix epoch of the most recent write
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS url_cache (
                    url TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """
            )
            # reports stores the final article/outline plus metadata per question:
            #   - question: original user question (primary key)
            #   - article: final article markdown
            #   - outline: outline markdown (may be empty string)
            #   - metadata: JSON blob with model/run info
            #   - updated_at: Unix epoch recording the most recent save
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reports (
                    question TEXT PRIMARY KEY,
                    article TEXT NOT NULL,
                    outline TEXT,
                    metadata TEXT,
                    updated_at INTEGER NOT NULL
                )
                """
            )

    def search_query_exists(self, query: str) -> bool:
        cur = self.conn.execute(
            "SELECT 1 FROM search_cache WHERE query = ? LIMIT 1", (query,)
        )
        return cur.fetchone() is not None

    def get_search_urls(self, query: str) -> list[str] | None:
        """
        Get the list of URLs for a given search query.
        Returns None if the query does not exist.
        """
        cur = self.conn.execute(
            "SELECT urls FROM search_cache WHERE query = ?", (query,)
        )
        row = cur.fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            logger.warning(f"Corrupted search_cache entry for query: {query}")
            return None

    def set_search_urls(self, query: str, urls: Iterable[str]) -> None:
        serialized = json.dumps(list(urls), ensure_ascii=False)
        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO search_cache(query, urls, updated_at)
                VALUES (?, ?, ?)
                """,
                (query, serialized, int(time.time())),
            )

    def has_url(self, url: str) -> bool:
        cur = self.conn.execute("SELECT 1 FROM url_cache WHERE url = ? LIMIT 1", (url,))
        return cur.fetchone() is not None

    def get_url_content(self, url: str) -> PageContent | None:
        cur = self.conn.execute("SELECT payload FROM url_cache WHERE url = ?", (url,))
        row = cur.fetchone()
        if not row:
            return None
        try:
            payload = json.loads(row[0])
            return PageContent(**payload)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning(f"Corrupted url_cache entry for {url}: {exc}")
            return None

    def set_url_content(self, url: str, page_content: PageContent) -> None:
        payload = page_content.model_dump(mode="json")
        serialized = json.dumps(payload, ensure_ascii=False)
        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO url_cache(url, payload, updated_at)
                VALUES (?, ?, ?)
                """,
                (url, serialized, int(time.time())),
            )

    def iter_url_items(self) -> Iterable[tuple[str, PageContent]]:
        cur = self.conn.execute("SELECT url, payload FROM url_cache")
        for url, payload in cur.fetchall():
            try:
                data = json.loads(payload)
                yield url, PageContent(**data)
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Skipping corrupted url_cache entry for {url}")

    def save_report(
        self,
        question: str,
        article: str,
        outline: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        serialized_meta = json.dumps(metadata or {}, ensure_ascii=False)
        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO reports(question, article, outline, metadata, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    question,
                    article,
                    outline,
                    serialized_meta,
                    int(time.time()),
                ),
            )

    def close(self) -> None:
        self.conn.close()
