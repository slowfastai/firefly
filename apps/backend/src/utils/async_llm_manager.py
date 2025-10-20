import asyncio
import aiohttp
from loguru import logger
from openai import AsyncOpenAI
from .report_cli import LLMResponse
from .helpers import is_o_or_gpt5


class LLMClient:
    """Base class for all async LLM clients
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

    async def chat_completion(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_completion_tokens: int | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: list[str] | None = None,
        timeout: int | None = None,
    ):
        """Call API with retry mechanism and timeout control - implemented in base class

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
            max_retries: Maximum number of retry attempts (default: 3)
            retry_delay: Delay between retries in seconds (default: 2)
            timeout: Custom request timeout for this call, if not set, will use default timeout

        Returns:
            str: The response from the API
        """
        effective_timeout = timeout if timeout else self.timeout
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
                return await asyncio.wait_for(
                    self._chat_completion(
                        message_list,
                        temperature,
                        top_p,
                        max_completion_tokens,
                        # top_k,  # OpenAI API does not support top_k
                        # repetition_penalty, # OpenAI API does not support repetition_penalty
                        frequency_penalty,
                        presence_penalty,
                        stop,
                    ),
                    timeout=effective_timeout,
                )

            except Exception as e:
                # Handle timeout specifically
                if isinstance(e, asyncio.TimeoutError):
                    logger.error(
                        f"[{self.__class__.__name__}] Model: {self.model_name} - Request timeout after {effective_timeout} seconds (attempt {attempt + 1}/{self.max_retries})"
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
                    await asyncio.sleep(wait_time)
                else:
                    error_type = (
                        "timeout" if isinstance(e, asyncio.TimeoutError) else "error"
                    )
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

    async def _chat_completion(
        self,
        message_list,
        temperature=None,
        top_p=None,
        max_completion_tokens=None,
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

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=message_list,
            **kwargs,
        )
        return LLMResponse(
            response_text=(response.choices[0].message.content or "").strip(),
            response_metadata={"usage": getattr(response, "usage", "")},
            actual_queried_message_list=message_list,
        )


class AsyncOpenAIClient(LLMClient):
    """
    OpenAI
    API reference: https://platform.openai.com/docs/api-reference/introduction
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
        super().__init__(
            model_name, api_key, base_url, timeout, max_retries, retry_delay
        )

    def _create_client(self):
        """Create OpenAI client instance"""
        return AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)


def create_llm_client(
    model_name,
    api_key,
    llm_base_url=None,
    timeout=15,
    max_retries=3,
    retry_delay=2,
):
    """Factory function to create corresponding LLM client instance based on model name"""
    # TODO: Extend beyond the current OpenAI-compatible clients (e.g., add Anthropic support).
    return AsyncOpenAIClient(
        model_name, api_key, llm_base_url, timeout, max_retries, retry_delay
    )
