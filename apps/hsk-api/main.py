# apps/hsk-api/main.py

from fastapi import FastAPI, Query
from rag.query import ask_question

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/ask")
def ask(q: str = Query(..., description="The question to ask the RAG model"),
        k: int = Query(4, description="Number of chunks to retrieve")):
    answer = ask_question(q, k=k)
    return {"question": q, "answer": answer}
