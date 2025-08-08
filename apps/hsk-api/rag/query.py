# apps/hsk-api/rag/query.py

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from rag.config import OPENAI_API_KEY, CHROMA_DB_DIR

# Load persisted vectorstore
def get_vectorstore():
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)

# Create a custom prompt for HSK-style responses
PROMPT_TEMPLATE = """
You are an expert Chinese language teacher specializing in HSK exams.
Use the provided context from documents to answer the user's question.
If possible, generate examples in the style of HSK questions.

Context:
{context}

Question:
{question}

Answer in the requested language.
"""

def ask_question(question: str, k: int = 4) -> str:
    # Load vectorstore
    db = get_vectorstore()
    retriever = db.as_retriever(search_kwargs={"k": k})

    # LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)

    # Prompt
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    # Chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa.run(question)
