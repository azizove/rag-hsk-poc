# Configuration file - to be implemented 
# apps/hsk-api/rag/config.py

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DOCS_DIR = os.path.join(os.path.dirname(__file__), "../docs")
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "../vectorstore")
