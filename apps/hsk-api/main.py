from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
from typing import Optional
from rag.query import ask_hsk

app = FastAPI(title="HSK RAG API")

class AskBody(BaseModel):
    q: str = Field(..., description="User request / instruction")
    type: str = Field("mcq", description="mcq | fill_blank | reading_comp | translation")
    level: int = Field(2, ge=1, le=9)
    difficulty: str = Field("medium", description="easy | medium | hard")
    numQuestions: int = Field(3, ge=1, le=10)
    k: int = Field(4, ge=1, le=12)
    outputLanguage: str = Field("English")

@app.get("/health")
def health_check():
    return {"status": "ok"}

# Keep simple GET for quick testing
@app.get("/ask")
def ask_get(
    q: str = Query(...),
    type: str = Query("mcq"),
    level: int = Query(2, ge=1, le=9),
    difficulty: str = Query("medium"),
    numQuestions: int = Query(3, ge=1, le=10),
    k: int = Query(4, ge=1, le=12),
    outputLanguage: str = Query("English"),
):
    result = ask_hsk(
        q,
        exercise_type=type,
        level=level,
        difficulty=difficulty,
        num_questions=numQuestions,
        output_language=outputLanguage,
        k=k,
    )
    return {"question": q, **result}

# Also provide a POST that’s nicer for the frontend
@app.post("/ask")
def ask_post(body: AskBody):
    result = ask_hsk(
        body.q,
        exercise_type=body.type,
        level=body.level,
        difficulty=body.difficulty,
        num_questions=body.numQuestions,
        output_language=body.outputLanguage,
        k=body.k,
    )
    return {"question": body.q, **result}
