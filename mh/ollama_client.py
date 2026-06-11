"""Minimal stdlib client for a local ollama server (no pip dependency).

Local, free, offline LLM inference. available()/model_present() let callers
fail closed gracefully when the server or model is absent — the MRAS critic
then falls back to its deterministic + NLI gates.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:3b"


def available(timeout: float = 2.0) -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=timeout)  # noqa: S310
        return True
    except (urllib.error.URLError, OSError):
        return False


def model_present(model: str = DEFAULT_MODEL, timeout: float = 3.0) -> bool:
    base = model.split(":")[0]
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=timeout) as r:  # noqa: S310
            tags = json.load(r)
    except (urllib.error.URLError, OSError):
        return False
    return any(m.get("name", "").startswith(base) for m in tags.get("models", []))


def generate_json(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    system: str | None = None,
    timeout: float = 120.0,
) -> dict:
    """Call /api/generate with JSON mode (temperature 0) and parse the reply."""
    body: dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }
    if system:
        body["system"] = system
    req = urllib.request.Request(  # noqa: S310
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
        resp = json.load(r)
    return json.loads(resp["response"])
