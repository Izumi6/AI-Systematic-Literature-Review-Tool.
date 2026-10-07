"""
llm/openai_client.py
LLM API client wrapper used by all analysis and synthesis modules.

Centralises model selection, provider configuration, error handling, retry logic,
and token management so callers only need to supply a prompt.
Supports OpenAI, Groq, DeepSeek, OpenRouter, and any OpenAI-compatible endpoint.
"""

import logging
import time
from typing import Optional

from openai import OpenAI, APIError, RateLimitError, APIConnectionError

from config import API_KEY, API_BASE_URL, LLM_MODEL, LLM_TEMPERATURE

logger = logging.getLogger(__name__)

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    """Return a module-level singleton API client."""
    global _client
    if _client is None:
        if not API_KEY:
            raise ValueError(
                "API key is not configured. Please enter your API key in the sidebar or set it in your .env file."
            )
        kwargs = {"api_key": API_KEY}
        if API_BASE_URL:
            kwargs["base_url"] = API_BASE_URL
        _client = OpenAI(**kwargs)
    return _client


def chat_completion(
    prompt: str,
    system_prompt: str = "You are a helpful academic research assistant.",
    max_tokens: int = 1000,
    temperature: float = LLM_TEMPERATURE,
    model: Optional[str] = None,
    retries: int = 3,
) -> Optional[str]:
    """
    Send a chat completion request and return the response text.

    Retries on rate-limit and connection errors with exponential backoff.
    Returns None on persistent failure.
    """
    selected_model = model or LLM_MODEL
    client = get_client()

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()

        except RateLimitError:
            wait = 2 ** attempt * 10
            logger.warning("Rate limit hit; waiting %d s (attempt %d).", wait, attempt + 1)
            time.sleep(wait)

        except APIConnectionError as exc:
            logger.warning("Connection error (attempt %d): %s", attempt + 1, exc)
            time.sleep(5)

        except APIError as exc:
            logger.error("LLM API error: %s", exc)
            break

    logger.error("All %d LLM attempts failed.", retries)
    return None
