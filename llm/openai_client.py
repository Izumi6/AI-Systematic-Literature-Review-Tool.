"""
llm/openai_client.py
LLM API client wrapper used by all analysis and synthesis modules.

Centralises model selection, provider configuration, error handling, retry logic,
and token management so callers only need to supply a prompt.
Supports Google Gemini, OpenAI, Groq, DeepSeek, OpenRouter, and any OpenAI-compatible endpoint.
"""

import logging
import time
from typing import Optional, Tuple

from openai import OpenAI, APIError, RateLimitError, APIConnectionError

import config

logger = logging.getLogger(__name__)

_client: Optional[OpenAI] = None
_cached_client_key: Optional[Tuple[str, str]] = None


def resolve_api_config(override_model: Optional[str] = None) -> Tuple[str, str, str]:
    """
    Dynamically resolve the active API key, base URL, and model from
    Streamlit session state, Streamlit secrets, config, or environment variables.
    """
    api_key = config.API_KEY or ""
    base_url = config.API_BASE_URL or ""
    model = override_model or config.LLM_MODEL or "gpt-4o-mini"

    # Check Streamlit session state and secrets if running in Streamlit
    try:
        import sys
        if "streamlit" in sys.modules:
            st = sys.modules["streamlit"]
            if hasattr(st, "runtime") and st.runtime.exists():
                if hasattr(st, "session_state"):
                    if st.session_state.get("runtime_key"):
                        api_key = str(st.session_state.runtime_key).strip()
                    elif st.session_state.get("api_key"):
                        api_key = str(st.session_state.api_key).strip()

                    if st.session_state.get("runtime_base_url"):
                        base_url = str(st.session_state.runtime_base_url).strip()
                    elif st.session_state.get("api_base_url"):
                        base_url = str(st.session_state.api_base_url).strip()

                    if not override_model:
                        if st.session_state.get("runtime_model"):
                            model = str(st.session_state.runtime_model).strip()
                        elif st.session_state.get("llm_model"):
                            model = str(st.session_state.llm_model).strip()

                if not api_key and hasattr(st, "secrets"):
                    for secret_name in ("API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"):
                        if secret_name in st.secrets:
                            api_key = str(st.secrets[secret_name]).strip()
                            break
    except Exception:
        pass

    # Clean key (strip trailing period or whitespace)
    api_key = api_key.strip().rstrip(".")

    # Auto-detect Google Gemini API key (starts with AIzaSy)
    if api_key.startswith("AIzaSy"):
        if not base_url:
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        if not model or model.lower() in ("gemini", "gpt-4o-mini", "default"):
            model = "gemini-2.5-flash"

    # Normalize common model names and aliases
    m_lower = model.lower()
    if m_lower in ("gemini", "gemini-flash", "gemini-1.5-flash"):
        model = "gemini-2.5-flash"
    elif m_lower in ("gemini-pro", "gemini-1.5-pro"):
        model = "gemini-2.5-pro"

    return api_key, base_url, model


def get_client() -> OpenAI:
    """Return a module-level singleton API client, auto-refreshing if credentials change."""
    global _client, _cached_client_key

    api_key, base_url, _ = resolve_api_config()
    if not api_key:
        raise ValueError(
            "API key is not configured. Please enter your API key in the sidebar or set it in your .env file."
        )

    client_key = (api_key, base_url)
    if _client is None or _cached_client_key != client_key:
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        _client = OpenAI(**kwargs)
        _cached_client_key = client_key

    return _client


def chat_completion(
    prompt: str,
    system_prompt: str = "You are a helpful academic research assistant.",
    max_tokens: int = 1500,
    temperature: float = config.LLM_TEMPERATURE,
    model: Optional[str] = None,
    retries: int = 3,
) -> Optional[str]:
    """
    Send a chat completion request and return the response text.

    Retries on rate-limit and connection errors with exponential backoff.
    Returns None on persistent failure.
    """
    api_key, base_url, selected_model = resolve_api_config(model)

    try:
        client = get_client()
    except Exception as exc:
        logger.error("Failed to initialize LLM client: %s", exc)
        return None

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
            wait = 2 ** attempt * 5
            logger.warning("Rate limit hit; waiting %d s (attempt %d).", wait, attempt + 1)
            time.sleep(wait)

        except APIConnectionError as exc:
            logger.warning("Connection error (attempt %d): %s", attempt + 1, exc)
            time.sleep(3)

        except APIError as exc:
            logger.error("LLM API error (%s): %s", selected_model, exc)
            break

        except Exception as exc:
            logger.error("Unexpected LLM error: %s", exc)
            break

    logger.error("All %d LLM attempts failed for model %s.", retries, selected_model)
    return None
