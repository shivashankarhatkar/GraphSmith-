import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

# USD per 1M tokens: (input, output). Placeholder prices; check each provider's pricing page.
PRICES = {
    "gemini-3.5-flash-lite": (0.10, 0.40),
    "claude-sonnet-4-5": (3.00, 15.00),
}


def model_id() -> str:
    model = os.getenv("MODEL")
    if not model:
        raise RuntimeError("Model name is not set in .env")
    return model


def model_name() -> str:
    """Model id without the provider prefix, e.g. 'anthropic:claude-x' -> 'claude-x'."""
    return model_id().split(":", 1)[-1]


def estimate_cost(name: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = PRICES.get(name, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def get_model(**overrides):
    model_settings = {
        "max_retries": 2,
        "timeout": 20,
        "max_tokens": int(os.getenv("MAX_TOKENS", 800)),
    }
    model_settings.update(overrides)
    return init_chat_model(model_id(), **model_settings)
