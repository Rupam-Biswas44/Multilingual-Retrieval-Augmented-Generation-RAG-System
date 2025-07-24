from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import numpy as np
import faiss
from pdf2image import convert_from_path
import pytesseract


def extract_text_via_ocr(pdf_path):
    print("[INFO] Extracting text from PDF via OCR...")
    pages = convert_from_path(pdf_path, dpi=300)
    all_text = []
    
    for i, page in enumerate(pages):
        print(f"[INFO] OCR processing page {i+1}/{len(pages)}...")
        text = pytesseract.image_to_string(page, lang='ben')  # Bangla language
        text = text.replace('\u200c', '').replace('\u200b', '').replace('\xa0', ' ')
        all_text.append(text.strip())
    
    return '\n'.join(all_text)


def load_and_split_pdf(pdf_path):
    raw_text = extract_text_via_ocr(pdf_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30,
        separators=["\n\n", "\n", "।", ".", "!", "?"]
    )
    texts = splitter.split_text(raw_text)
    return [Document(page_content=t) for t in texts]


def build_vectorstore(chunks, openai_api_key):
   
    embeddings = OpenAIEmbeddings(
        openai_api_key=openai_api_key,
        model="text-embedding-3-small"
    )
    
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    vector_list = embeddings.embed_documents(texts)
    vector_array = np.array(vector_list).astype("float32")
    index = faiss.IndexFlatL2(vector_array.shape[1])
    index.add(vector_array)
    return index, vector_array, texts, metadatas, embeddings