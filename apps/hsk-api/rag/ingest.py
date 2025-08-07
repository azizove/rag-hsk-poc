# apps/hsk-api/rag/ingest.py

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from rag.config import OPENAI_API_KEY, DOCS_DIR, CHROMA_DB_DIR
from rag.ocr_utils import ocr_pdf
import os

def ingest_documents():
    print("🔍 Scanning folder for PDFs...")
    all_docs = []

    for filename in os.listdir(DOCS_DIR):
        if not filename.endswith(".pdf"):
            continue

        filepath = os.path.join(DOCS_DIR, filename)
        print(f"📄 Processing: {filename}")
        loader = PyPDFLoader(filepath)

        try:
            docs = loader.load()
            if not docs or all(doc.page_content.strip() == "" for doc in docs):
                raise ValueError("No text found. Trying OCR fallback.")
            all_docs.extend(docs)
            print(f"✅ Extracted text from {filename} using PyPDFLoader")
        except Exception as e:
            print(f"⚠️  {e} -- Using OCR for {filename}")
            ocr_docs = ocr_pdf(filepath)
            all_docs.extend(ocr_docs)
            print(f"✅ OCR completed for {filename}")

    print(f"📄 Total documents: {len(all_docs)}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)
    print(f"✂️ Chunks created: {len(chunks)}")

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_DB_DIR)
    db.persist()
    print("✅ Vector DB saved at:", CHROMA_DB_DIR)
