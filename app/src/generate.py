import json
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

# local development only: load the repo-root .env if it exists. It never overrides
# real environment variables, and in Docker/Azure GROQ_API_KEY comes from the runtime.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

client = Groq()  # reads GROQ_API_KEY from the environment

MODEL = "openai/gpt-oss-120b"


def generate_answer(prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "used_context": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["answer", "used_context"],
    "additionalProperties": False,
}


def generate_structured(prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "rag_answer", "strict": True, "schema": ANSWER_SCHEMA},
        },
    )
    return json.loads(response.choices[0].message.content)
