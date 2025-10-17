import requests
import json
import time
import threading
from typing import Any
from openai import OpenAI
from loguru import logger
from report_cli import LLMResponse
from helpers import is_o_or_gpt5


class TimeoutError(Exception):
    """Custom timeout exception for synchronous calls"""

    pass


class LLMClient:
    """Base class for all LLM clients
    1. for typing hint
    2. implement chat_completion method in base class

    Args:
        model_name: str, the name of the model
        api_key: str, the API key for the model
        base_url: str, the base URL for the model
    Attributes:
        model_name: str, the name of the model
        api_key: str, the API key for the model
    """

    def __init__(
        self,
        model_name,
        api_key,
        base_url=None,
        timeout=15,
        max_retries=3,
        retry_delay=2,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = self._create_client()

    def _create_client(self):
        """Create client instance"""
        raise NotImplementedError("Subclasses must implement this method")

    def chat_completion(
        self,
        prompt,
        system_prompt=None,
        temperature=None,
        top_p=None,
        max_completion_tokens=None,
        top_k=None,
        repetition_penalty=None,
        frequency_penalty=None,
        presence_penalty=None,
        stop=None,
    ):
        """Call OpenAI API with retry mechanism and timeout control - implemented in base class

        Args:
            prompt: The input prompt
            system_prompt: Optional system prompt
            temperature: Optional temperature parameter
            top_p: Optional top_p parameter
            max_completion_tokens: Optional max_completion_tokens parameter
            top_k: Optional top_k parameter
            repetition_penalty: Optional repetition_penalty parameter
            frequency_penalty: Optional frequency_penalty parameter
            presence_penalty: Optional presence_penalty parameter
            stop: Optional stop sequences
        """
        message_list = (
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            if system_prompt is not None
            else [{"role": "user", "content": prompt}]
        )
        last_error = ""
        for attempt in range(self.max_retries):
            try:
                # Create a timeout mechanism using threading
                result_container: dict[str, Any] = {"result": None, "exception": None}

                def api_call():
                    try:
                        result_container["result"] = self._chat_completion(
                            message_list,
                            temperature,
                            top_p,
                            max_completion_tokens,
                            # top_k,  # OpenAI API does not support top_k
                            # repetition_penalty, # OpenAI API does not support repetition_penalty
                            frequency_penalty,
                            presence_penalty,
                            stop,
                        )
                    except Exception as e:
                        result_container["exception"] = e

                # Start API call in a separate thread
                api_thread = threading.Thread(target=api_call)
                api_thread.daemon = True
                api_thread.start()

                # Wait for completion or timeout
                api_thread.join(timeout=self.timeout)

                if api_thread.is_alive():
                    # Timeout occurred
                    raise TimeoutError(f"Request timeout after {self.timeout} seconds")

                if result_container["exception"]:
                    raise result_container["exception"]

                return result_container["result"]

            except Exception as e:
                # Handle timeout specifically
                if isinstance(e, TimeoutError):
                    logger.error(
                        f"[{self.__class__.__name__}] Model: {self.model_name} - Request timeout after {self.timeout} seconds (attempt {attempt + 1}/{self.max_retries})"
                    )
                # Handle other errors
                else:
                    logger.error(
                        f"[{self.__class__.__name__}] Model: {self.model_name} - Encountered error (attempt {attempt + 1}/{self.max_retries})... Error: {e}"
                    )

                # Retry logic for all error types (except 503)
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (attempt + 1)
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    error_type = "timeout" if isinstance(e, TimeoutError) else "error"
                    last_error = f"Reached maximum retry attempts ({self.max_retries}), giving up due to {error_type}"
                    logger.error(last_error)
                    return LLMResponse(
                        response_text="",
                        response_metadata={
                            "usage": "",
                            "error": last_error,
                            "attempts": self.max_retries + 1,
                        },
                        actual_queried_message_list=message_list,
                    )

    def _chat_completion(
        self,
        message_list,
        temperature=None,
        top_p=None,
        max_completion_tokens=None,
        # top_k=None, # OpenAI API does not support top_k
        # repetition_penalty=None, # OpenAI API does not support repetition_penalty
        frequency_penalty=None,
        presence_penalty=None,
        stop=None,
        reasoning_effort=None,
    ):
        """
        OpenAI compatible API
        https://platform.openai.com/docs/api-reference/introduction
        """
        kwargs: dict[str, object] = {}
        if is_o_or_gpt5(self.model_name):
            if max_completion_tokens is not None:
                kwargs["max_completion_tokens"] = max_completion_tokens
            if reasoning_effort is not None:
                kwargs["reasoning_effort"] = reasoning_effort
        else:
            if max_completion_tokens is not None:
                # TODO maybe other OpeenAI-compatible API alos supports `max_completion_tokens`?
                kwargs["max_tokens"] = max_completion_tokens
            if temperature is not None:
                kwargs["temperature"] = temperature

            if reasoning_effort is not None:
                # TODO maybe other OpeenAI-compatible API alos supports `reasoning_effort`?
                kwargs["reasoning"] = {"effort": reasoning_effort}
            if top_p is not None:
                kwargs["top_p"] = top_p
            if stop is not None:
                kwargs["stop"] = stop
        if frequency_penalty is not None:
            kwargs["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            kwargs["presence_penalty"] = presence_penalty

        # if (
        #         self.llm_config.base_url is not None
        #         and "openrouter.ai" in self.llm_config.base_url
        #         and self.args.openrouter_usage
        #     ):
        #         kwargs["usage"] = {"include": self.args.openrouter_usage}

        # Add provider-specific headers if needed
        # headers = self._get_openai_headers()
        # if headers:
        #     extra_params["extra_headers"] = headers

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=message_list,
            **kwargs,
        )
        return LLMResponse(
            response_text=(response.choices[0].message.content or "").strip(),
            response_metadata={"usage": getattr(response, "usage", "")},
            actual_queried_message_list=message_list,
        )


class OpenAIClient(LLMClient):
    """OpenAI"""

    def __init__(
        self,
        model_name,
        api_key,
        base_url=None,
        timeout=15,
        max_retries=3,
        retry_delay=2,
    ):
        super().__init__(
            model_name, api_key, base_url, timeout, max_retries, retry_delay
        )

    def _create_client(self):
        """Create OpenAI client instance"""
        return OpenAI(api_key=self.api_key, base_url=self.base_url)


def create_llm_client(
    model_name,
    api_key,
    llm_base_url=None,
    timeout=15,
    max_retries=3,
    retry_delay=2,
):
    """Factory function to create corresponding LLM client instance based on model name"""
    return OpenAIClient(
        model_name, api_key, llm_base_url, timeout, max_retries, retry_delay
    )
