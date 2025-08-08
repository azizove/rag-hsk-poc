# apps/hsk-api/rag/query.py

from typing import Dict, Any, List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from rag.config import OPENAI_API_KEY, CHROMA_DB_DIR
from rag.prompts import build_prompt

def get_vectorstore():
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)

def _normalize_query(q: str) -> str:
    """Normalize user queries so HSKK is treated as HSK for retrieval & generation."""
    q = q.replace("HSKK", "HSK").replace("hskk", "hsk")
    # Extra nudge to the model baked into the user question:
    q += "\n\nNote: Treat HSKK and HSK as equivalent."
    return q

def dedupe_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique = []
    for s in sources:
        key = (s.get("source"), s.get("page"))
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique

def ask_hsk(
    question: str,
    *,
    exercise_type: str = "mcq",
    level: int = 2,
    difficulty: str = "medium",
    num_questions: int = 3,
    output_language: str = "English",
    k: int = 6,  # ↑ a bit to broaden recall
) -> Dict[str, Any]:
    db = get_vectorstore()
    retriever = db.as_retriever(search_kwargs={"k": k})

    llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)

    prompt = build_prompt(
        exercise_type=exercise_type,
        level=level,
        difficulty=difficulty,
        num_questions=num_questions,
        output_language=output_language,
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    normalized_q = _normalize_query(question)
    result = qa({"query": normalized_q})

    answer: str = result["result"]
    sources = []
    for doc in result.get("source_documents", []):
        meta = doc.metadata or {}
        sources.append(
            {
                "source": meta.get("source"),
                "page": meta.get("page"),
                "score": meta.get("score"),
            }
        )

    return {
        "answer": answer,
        "exercise_type": exercise_type,
        "level": level,
        "difficulty": difficulty,
        "num_questions": num_questions,
        "output_language": output_language,
        "k": k,
        "sources": dedupe_sources(sources),
    }
