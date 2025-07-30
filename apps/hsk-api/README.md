# HSK RAG API - Step 1: Basic Setup

This is the first step in building the HSK RAG API. We're starting with a minimal FastAPI application.

## Current Structure

```
apps/hsk-api/
│
├── main.py              # Basic FastAPI entry point
├── rag/                 # RAG package (empty files)
│   ├── __init__.py      # Empty
│   ├── config.py        # Empty
│   ├── ingest.py        # Empty
│   └── query.py         # Empty
├── vectorstore/         # Empty directory
├── docs/                # Empty directory
├── requirements.txt     # Minimal dependencies
└── README.md           # This file
```

## Setup

1. **Install basic dependencies first:**
   ```bash
   cd apps/hsk-api
   pip install fastapi uvicorn
   ```

2. **Test the basic setup:**
   ```bash
   python main.py
   ```

## Next Steps

We'll add dependencies gradually to avoid build issues:
1. ✅ Basic FastAPI setup
2. 🔄 Add configuration management
3. 🔄 Add file upload functionality
4. 🔄 Add PDF processing
5. 🔄 Add vector store integration
6. 🔄 Add RAG query functionality 