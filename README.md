# Multilingual RAG System (English & Bengali)

This project implements a lightweight Retrieval-Augmented Generation (RAG) system capable of answering English and Bengali questions . It supports long-term memory from documents and short-term conversational memory.

## 💡 Features

* 🔍 RAG pipeline with OpenAI & FAISS
* 🧠 Long-Term Memory: HSC textbook vectorized
* 🧠 Short-Term Memory: Conversational history per session
* 🚩 Multilingual Query Support (Bangla + English)
* 🗞️ Bengali OCR and cleaning
* 🌐 RESTful API via FastAPI

---

## 🚀 Setup Guide


1. Create virtual environment and install dependencies:

   ```bash
   python -m venv rag_env
   source rag_env/bin/activate
   pip install -r requirements.txt
   ```

2. Install required system dependencies:

   * poppler: for pdf2image
     macOS: `brew install poppler`
   * Tesseract with Bengali support:

     ```bash
     brew install tesseract
     brew install tesseract-lang
     ```

3. Run the RAG backend:

   ```bash
   python main.py
   ```
4. Folder setup :
├── app
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-311.pyc
│   │   ├── __init__.cpython-38.pyc
│   │   ├── api.cpython-311.pyc
│   │   ├── api.cpython-38.pyc
│   │   ├── rag_pipeline.cpython-311.pyc
│   │   ├── rag_pipeline.cpython-38.pyc
│   │   └── utils.cpython-311.pyc
│   ├── api.py
│   ├── rag_pipeline.py
│   └── utils.py
├── data
│   └── HSC26-Bangla1st-Paper.pdf
├── main.py
├── outputs
├── README.md
├── requirements.txt
└── ui
    └── app_ui.py
---

## 💠 Tools & Libraries Used

| Component  | Tool / Library                    |
| ---------- | --------------------------------- |
| LLM        | OpenAI GPT-3.5 Turbo              |
| Embeddings | text-embedding-3-small            |
| Vector DB  | FAISS                             |
| OCR        | pytesseract, pdf2image            |
| API        | FastAPI                           |
| Chunking   | RecursiveCharacterTextSplitter    |
| Memory     | Short (chat history), Long (docs) |

---

## 📙 Sample Queries & Outputs

| Query (Bengali)                                | Expected Answer | Result |
| ---------------------------------------------- | --------------- | ------ |
| অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?          | শুম্ভুনাথ       | ✅      |  ✅
| কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে? | মামাকে         | ✅      |  ✅
| বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?       | 15 বছর         | ✅     |  ✅ 

---

## 📊 Evaluation Matrix

| Metric       | Description                                                             | Implementation                                            |
| ------------ | ----------------------------------------------------------------------- | --------------------------------------------------------- |
| Groundedness | Answer based on retrieved chunks only                                   | System filters context → prompt enforces no-hallucination |
| Relevance    | Uses MMR retrieval (max marginal relevance) to improve contextual match | ✅ FAISS MMR k=10                                          |
| Short Memory | Conversational history (optional, ready to integrate)                   | Per-session storage                                       |
| Long Memory  | Full HSC book vectorized using OCR + FAISS                              | ✅ Done                                                    |

---

## 🔍 RAG Design Q\&A

### 1. What method/library did I use to extract the text and why?

We used pytesseract (OCR) with pdf2image since the HSC PDF had embedded images, not extractable text. This gave us full control over Bangla text extraction, even with low-quality formatting.

### 2. What chunking strategy did I use?

RecursiveCharacterTextSplitter with chunk\_size=300, overlap=30. We chose character-based chunking to maintain linguistic boundaries in Bengali which avoids breaking semantic meaning mid-sentence or mid-word.

### 3. What embedding model did I use and why?

We used OpenAI’s text-embedding-3-small. It provides fast, multilingual support including Bangla, and is optimized for retrieval performance with good cost efficiency.

### 4. How am I comparing the query with my stored chunks?

We use cosine similarity over FAISS with L2 index and MMR (Max Marginal Relevance) search to balance relevance and diversity.

### 5. How do I ensure meaningful comparison between question and document chunks?

All text is normalized, chunked meaningfully, and embedded using the same model. We use MMR and post-filter short or gibberish chunks. Prompting enforces grounded answer strictly from context.

### 6. Do the results seem relevant? If not, how could it improve?

✅ Relevance is high due to clean OCR and proper chunking. Further improvements:

* Better OCR language models (e.g., Bengali Tesseract finetuned)
* Larger chunk size for semantic completeness
* Hybrid reranking (e.g., re-rank retrieved results using LLM)

---

## 🌐 API Documentation

Endpoint: POST /query

Request:

```json
{
  "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
}
```

Response:

```json
{
  "answer": "শুম্ভুনাথ"
}
```

---

## ✅ Bonus

✅ Lightweight REST API
✅ Multilingual Queries
✅ Evaluation Matrix (Groundedness + Relevance)
✅ Stramlit basic frontend


---

## 📌 Notes

* Vector DB is in-memory FAISS 
* All processing is local + OpenAI API
* Prompt is tuned to disallow hallucination
