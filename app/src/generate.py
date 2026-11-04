import json
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# local development only: load the repo-root .env if it exists. It never overrides
# real environment variables, and in Docker/Azure GROQ_API_KEY comes from the runtime.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "used_context": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["answer", "used_context"],
    "additionalProperties": False,
}


@lru_cache(maxsize=1)
def get_client():
    from groq import Groq

    # Fail fast: a visitor should get the fallback message, not a 60 s hang.
    return Groq(timeout=20.0, max_retries=1)  # reads GROQ_API_KEY from the environment


def generate_structured(prompt):
    response = get_client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "rag_answer", "strict": True, "schema": ANSWER_SCHEMA},
        },
    )
    return json.loads(response.choices[0].message.content)
