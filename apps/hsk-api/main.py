from fastapi import FastAPI
# from rag.query import ask_question

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/ask")
def ask(q: str):
    return {"answer": "This is aziz basha"}
