"""
llm/openai_client.py
OpenAI API client wrapper used by all LLM-dependent modules.

Centralises model selection, error handling, retry logic, and
token management so callers only need to supply a prompt.
"""

import logging
import time
from typing import Optional

from openai import OpenAI, APIError, RateLimitError, APIConnectionError

from config import OPENAI_API_KEY, OPENAI_MODEL, LLM_TEMPERATURE

logger = logging.getLogger(__name__)

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    """Return a module-level singleton OpenAI client."""
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set. Please add it to your .env file."
            )
        _client = OpenAI(api_key=OPENAI_API_KEY)
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
    selected_model = model or OPENAI_MODEL
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
            logger.error("OpenAI API error: %s", exc)
            break

    logger.error("All %d LLM attempts failed.", retries)
    return None
