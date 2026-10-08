from typing import Annotated

from fastapi import FastAPI
from pydantic import BaseModel, StringConstraints
from src.rag import answer_question
from fastapi.middleware.cors import CORSMiddleware
from src.database import container




app = FastAPI()


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


@app.get("/")
def root():
    return {
        "status": "Ask Bach API is running"
    }



@app.post("/ask")
def ask(request: QuestionRequest):
    return answer_question(request.question)

@app.get("/content")
def get_content():
    items = list(
        container.query_items(
            query="""
                SELECT c.id, c.source, c.text
                FROM c
                WHERE c.type = 'document'
            """,
            enable_cross_partition_query=True,
        )
    )

    return {"documents": items}

@app.get("/content/experience")
def get_experience():
    items = list(
        container.query_items(
            query="""
                SELECT
                    c.id,
                    c.organization,
                    c.role,
                    c.period,
                    c.highlights,
                    c.technologies
                FROM c
                WHERE c.type = 'experience'
                AND c.isPublic = true
            """,
            partition_key="experience",
        )
    )

    return {"experiences": items}