from fastapi import FastAPI
from pydantic import BaseModel
from app.rag import get_answer

app = FastAPI()

class Query(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/ask")
def ask(query: Query):
    answer = get_answer(query.question)

    return {
        "question": query.question,
        "answer": answer
    }
