"""
Send a prompt to the local LLM (Ollama) and get its answer.

Uses only the Python standard library, so no extra package is needed.
"""

import json
import urllib.error
import urllib.request

from src.config import LLM_CONTEXT_TOKENS, LLM_MODEL, OLLAMA_URL


def ask_llm(prompt: str, model: str = LLM_MODEL) -> str:
    """
    Send one prompt and return the model's reply as text.
    """

    # 1. Build the request
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,              # wait for the full answer
        "options": {
            "temperature": 0,         # same question -> same answer
            "num_ctx": LLM_CONTEXT_TOKENS
        }
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    # 2. Send it and read the reply
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            reply = json.loads(response.read())

    except urllib.error.URLError as error:
        raise ConnectionError(
            "Could not reach Ollama. Is it running? Try: ollama serve"
        ) from error

    return reply["response"].strip()
