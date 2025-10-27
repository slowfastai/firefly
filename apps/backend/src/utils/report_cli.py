import os
import sys
import time
import random
import argparse
import datetime
from typing import Any
from collections import namedtuple
from dataclasses import dataclass, field

import numpy as np
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Run DeepResearch")
    parser.add_argument(
        "--single_question",
        type=str,
        default=None,
        help="Single question to process.",
    )
    parser.add_argument(
        "--reasoning_model_name",
        type=str,
        default=None,
        help="Name of the reasoning model that orchestrates the deepresearch workflow.",
    )
    parser.add_argument(
        "--writer_model_name",
        type=str,
        default=None,
        help="Name of the model used to write or edit article/report.",
    )
    parser.add_argument(
        "--summarizer_model_name",
        type=str,
        default=None,
        help="Name of the model used to summarize fetched web content.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature."
    )
    parser.add_argument(
        "--top_p", type=float, default=0.8, help="Top-p sampling parameter."
    )
    parser.add_argument(
        "--min_p", type=float, default=0.05, help="Minimum p sampling parameter."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k sampling parameter. Warning: OpenAI API does not support top_k.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=None,
        help="Repetition penalty. Warning: OpenAI API does not support repetition_penalty.",
    )
    parser.add_argument(
        "--frequency_penalty",
        type=float,
        default=None,
        help="Frequency penalty used in OpenAI API.",
    )
    parser.add_argument(
        "--presence_penalty",
        type=float,
        default=None,
        help="Presence penalty used in OpenAI API.",
    )
    parser.add_argument(
        "--max_completion_tokens",
        type=int,
        default=None,
        help="An upper bound for the number of tokens that can be generated for a chat completion, \
              including visible output tokens and reasoning tokens.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=90,
        help="Request timeout in seconds for LLM API calls. If an API call takes longer than this value, it will be considered failed and retried. Default: 20 seconds.",
    )
    parser.add_argument(
        "--search_plan_timeout",
        type=int,
        default=90,
        help="Timeout in seconds specifically for the initial search-plan LLM call, which may require longer thinking. Overrides --timeout only for that step. Default: 40",
    )
    parser.add_argument(
        "--max_tool_calls", type=int, default=5, help="Maximum number of tool calling."
    )
    parser.add_argument(
        "--max_plan_steps",
        type=int,
        default=8,
        help="Maximum of number search planning steps.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum number of retry attempts for LLM API calls when encountering errors or timeouts. Default: 3",
    )
    parser.add_argument(
        "--retry_delay",
        type=int,
        default=2,
        help="Delay in seconds between retry attempts for LLM API calls. The delay increases exponentially: 2s, 4s, 6s, etc. Default: 2",
    )

    parser.add_argument(
        "--max_search_results",
        type=int,
        default=10,
        help="Maximum number of search documents to return.",
    )
    parser.add_argument(
        "--keep_links",
        action="store_true",
        default=False,
        help="Whether to keep links in fetched web content",
    )
    parser.add_argument(
        "--is_using_deep_web_explore",
        action="store_true",
        default=False,
        help="Whether to use deep web explore mode",
    )
    parser.add_argument(
        "--extractor",
        type=str,
        default="crawl4ai",
        choices=["crawl4ai", "requests", "jina"],
        help="Web content fetching method. Default: crawl4ai",
    )
    parser.add_argument(
        "--browser_type",
        type=str,
        default="chrome",
        choices=[
            "chrome",
            "chromium",
            "firefox",
            "edge",
            "safari",
            "brave",
            "opera",
            "vivaldi",
            "whale",
        ],
        help="Browser type to use for web browsing, used for getting user agent and cookies",
    )
    parser.add_argument(
        "--use_selenium",
        action="store_true",
        help="Whether to use selenium to get user agent",
    )
    parser.add_argument(  # TODO support real search engine apis, like serper; support jina api for crawl web page
        "--search_engine",
        type=str,
        default="cookie_google",
        choices=[
            # "serper",
            "cookie_google",
            "duckduckgo",
            "startpage",
            "brave",
            "bing",
            # "cookie_bing",
            # "cookie_duckduckgo",
            # "cookie_baidu",
            # "cookie_yandex",
        ],
        help="Search engine to use, 'cookie' represents not to call search engine api. \
            Default: cookie_google",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Whether to run evaluation after generation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for generation. If not set, will use current timestamp as seed.",
    )
    parser.add_argument(
        "--concurrent_limit",
        type=int,
        default=32,
        help="Maximum number of concurrent API calls",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="/share/project/llm/QwQ-32B",
        help="Path to the main tokenizer",
    )
    parser.add_argument(
        "--aux_tokenizer_path",
        type=str,
        default="/share/project/llm/Qwen2.5-32B-Instruct",
        help="Path to the auxiliary tokenizer",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="Path to the log directory",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Set log file path for loguru logger.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="Path to the cache directory",
    )
    return parser.parse_args()


def get_logger(log_dir, log_file=None):
    if log_file is None:
        log_file = (
            log_dir
            + "/"
            + (
                datetime.datetime.now().strftime(
                    "webthinker_report" + "_%Y%m%d%H%M%S" + ".log"
                )
            )
        )
    else:
        log_file = (
            log_dir
            + "/"
            + (datetime.datetime.now().strftime(f"{log_file}_%Y%m%d%H%M%S.log"))
        )

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Remove default handler
    logger.remove()
    # Add console output unless streaming mode is on (desktop event stream)
    if os.getenv("DR_STREAMING", "0") != "1":
        logger.add(
            sys.stdout,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
            level="INFO",
        )
    # Add file output
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
        level="INFO",
        rotation="100 MB",
        retention="7 days",
    )

    logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    return logger


def set_seed(seed):
    if seed is None:
        seed = int(time.time())
    random.seed(seed)
    np.random.seed(seed)
    return seed


TokenUsage = namedtuple(
    "TokenUsage",
    [
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "response_tokens",
    ],
)

Message = dict[str, Any]  # keys role, content
MessageList = list[Message]


@dataclass
class LLMResponse:
    """
    Response from calling OpenAI chat completion api.
    """

    response_text: str
    response_metadata: dict[str, Any]
    actual_queried_message_list: MessageList
