"""
Shared code for every lesson: which Azure resource to call, which deployments
to use, and small helpers that send a chat or an embeddings request.

The same helpers as azure_client.py in AzureAI_Foundry (chat, chat_raw, embed),
calling the class's Azure OpenAI deployments through the official openai
package and the v1 API:  {endpoint}/openai/v1/

    from azure_client import chat, embed, setting

All values come from the .env file next to this script. It is the SAME file as in
AzureAI_Foundry\\py: copy it across (trainer\\setup_rag.ps1 does this for the trainer).
Never commit .env.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletion

LESSON_DIR = Path(__file__).resolve().parent
DATA_DIR = LESSON_DIR.parent / "data"

load_dotenv(LESSON_DIR / ".env")


def setting(name: str, default: str | None = None) -> str:
    """Read one value from .env (or an environment variable) and stop with a clear message if it is missing."""
    value = os.getenv(name, default)
    if not value:
        sys.exit(f"Missing {name}. Copy .env.example to .env and fill in the values from your trainer.")
    return value


ENDPOINT = setting("AZURE_OPENAI_ENDPOINT").rstrip("/")     # https://<resource>.openai.azure.com
API_KEY = setting("AZURE_OPENAI_API_KEY")
CHAT_DEPLOYMENT = setting("AZURE_OPENAI_CHAT_DEPLOYMENT", "chat")     # the deployment name, not the model name
EMBED_DEPLOYMENT = setting("AZURE_OPENAI_EMBED_DEPLOYMENT", "embed")

# gpt-5.x models think before they answer. "none" switches that off: faster and cheaper,
# and plenty for answering from sources. Leave it empty for a model that does not take
# this setting (for example gpt-4.1-mini).
REASONING_EFFORT = os.getenv("AZURE_OPENAI_REASONING_EFFORT", "none")

# The v1 API: the standard OpenAI client, pointed at your Azure resource.
# No api-version to pick; "model" in every request is your DEPLOYMENT name.
BASE_URL = f"{ENDPOINT}/openai/v1/"
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def _with_defaults(options: dict) -> dict:
    if REASONING_EFFORT and "reasoning_effort" not in options:
        options = {"reasoning_effort": REASONING_EFFORT, **options}
    return options


def chat_raw(messages: list[dict], response_format: dict | None = None, **options) -> ChatCompletion:
    """Send a list of messages and return the full reply object (with token counts)."""
    if response_format is not None:
        options["response_format"] = response_format
    return client.chat.completions.create(model=CHAT_DEPLOYMENT, messages=messages, **_with_defaults(options))


def chat(messages: list[dict], response_format: dict | None = None, **options) -> str:
    """Send a list of messages and return only the model's answer text."""
    return chat_raw(messages, response_format, **options).choices[0].message.content or ""


def embed(texts: list[str], dimensions: int | None = None) -> list[list[float]]:
    """Turn a list of texts into a list of embeddings, one per text, in ONE request.

    dimensions:  text-embedding-3 models can return shorter vectors (lesson 1). None = the full 1,536.
    """
    options = {"dimensions": dimensions} if dimensions else {}
    reply = client.embeddings.create(model=EMBED_DEPLOYMENT, input=texts, **options)
    return [item.embedding for item in reply.data]
