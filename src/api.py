from typing import Annotated

from fastapi import FastAPI
from pydantic import BaseModel, StringConstraints
from src.rag import answer_question
from fastapi.middleware.cors import CORSMiddleware




app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://bachnguyennn.github.io",
    ],
    allow_credentials=False,
    allow_methods=["POST"],
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
