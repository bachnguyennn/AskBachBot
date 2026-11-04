import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, StringConstraints

from src import rag
from src.rate_limit import RateLimiter

# Root stays at WARNING so the Azure SDK doesn't log every HTTP request; the app's
# own logger reports warm-up time and per-request latency and sources at INFO.
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("ask_bach")
log.setLevel(logging.INFO)

FALLBACK_ANSWER = (
    "Sorry, I can't answer right now. Please try again in a minute, "
    "or reach Bach directly through the contact links on this site."
)

ask_limiter = RateLimiter(
    max_requests=int(os.getenv("RATE_LIMIT_PER_MINUTE", "10")), window_seconds=60
)


@asynccontextmanager
async def lifespan(_app):
    # Warm up at startup (embedding model, Cosmos DB documents, BM25 index) so the
    # first visitor doesn't pay for it. A failure here is logged, not fatal: the
    # app still starts and /ask retries the load on the next request.
    start = time.perf_counter()
    try:
        from src.dense_retriever import get_model
        from src.hybrid_retriever import get_retriever

        get_model()
        retriever = get_retriever()
        log.info("warmup ok: %d chunks in %.1fs", len(retriever.chunks), time.perf_counter() - start)
    except Exception:
        log.exception("warmup failed; will retry on first request")
    yield


app = FastAPI(title="Ask Bach API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://bachnguyennn.github.io",
        "http://localhost:4321",
        "http://127.0.0.1:4321",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class QuestionRequest(BaseModel):
    # strip first, then check length, so "     " becomes "" and fails min_length
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]


def client_ip(request: Request) -> str:
    # Azure Container Apps ingress sits in front of the app, so the visitor's
    # address is the first hop in X-Forwarded-For.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit(request: Request):
    allowed, retry_after = ask_limiter.hit(client_ip(request))
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Too many questions. Please wait a moment and try again.",
            headers={"Retry-After": str(retry_after)},
        )


@app.get("/")
def root():
    return {"status": "Ask Bach API is running"}


@app.post("/ask", dependencies=[Depends(rate_limit)])
def ask(request: QuestionRequest):
    start = time.perf_counter()
    try:
        response = rag.answer_question(request.question)
    except Exception:
        # Groq or Cosmos DB failed. Return a clean, retryable error with a message
        # the widget can show, instead of a raw 500.
        log.exception("ask failed after %.2fs", time.perf_counter() - start)
        return JSONResponse(
            status_code=503,
            content={"answer": FALLBACK_ANSWER, "sources": [], "error": "upstream_unavailable"},
        )
    log.info(
        "ask ok latency=%.2fs sources=%s",
        time.perf_counter() - start,
        [f"{s['source']}#{s['chunk_id']}" for s in response["sources"]],
    )
    return response


def query_content(query, **kwargs):
    from src.database import get_container

    try:
        return list(get_container("content").query_items(query=query, **kwargs))
    except Exception as err:
        log.exception("content query failed")
        raise HTTPException(status_code=503, detail="Content store unavailable.") from err


@app.get("/content")
def get_content():
    items = query_content(
        "SELECT c.id, c.source, c.text FROM c WHERE c.type = 'document'",
        enable_cross_partition_query=True,
    )
    return {"documents": items}


@app.get("/content/experience")
def get_experience():
    items = query_content(
        """
        SELECT c.id, c.organization, c.role, c.period, c.highlights, c.technologies
        FROM c
        WHERE c.type = 'experience' AND c.isPublic = true
        """,
        partition_key="experience",
    )
    return {"experiences": items}
