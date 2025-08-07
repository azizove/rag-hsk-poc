# apps/hsk-api/rag/ocr_utils.py

import pytesseract
from pdf2image import convert_from_path
from langchain.schema import Document

def ocr_pdf(path: str) -> list[Document]:
    images = convert_from_path(path)
    docs = []
    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image)
        docs.append(Document(
            page_content=text,
            metadata={"source": path, "page": i}
        ))
    return docs
