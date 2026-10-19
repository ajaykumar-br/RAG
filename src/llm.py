"""One LLM call, wrapped. Everything else in the project calls complete().

Setup (Git Bash):  export OPENROUTER_API_KEY="sk-or-..."     (never put the key in code or git)
Test:              uv run python src/llm.py "Say hello in five words."
"""
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv(override=True)   # finds .env by searching upward from this file's folder

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = os.environ.get("LLM_MODEL", "~anthropic/claude-sonnet-latest")


def complete(user, system=None, temperature=0.0, max_tokens=500, model=None):
    """Send one request. Returns (text, usage)."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY first (see the top of this file).")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    payload = {"model": model or MODEL, "messages": messages, "max_tokens": max_tokens}
    if temperature is not None:
        payload["temperature"] = temperature

    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    if not response.ok:
        raise RuntimeError(f"OpenRouter error {response.status_code}: {response.text[:500]}")
    data = response.json()
    return data["choices"][0]["message"]["content"], {**data.get("usage", {}), "model": data.get("model")}


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or "Say hello in five words."
    text, usage = complete(prompt)
    print(text)
    print("usage:", usage)