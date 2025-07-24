from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from app.rag_pipeline import SimpleRAG

app = FastAPI()

rag = SimpleRAG("data/HSC26-Bangla1st-Paper.pdf")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
class QueryInput(BaseModel):
    query: str
@app.post("/query")
def ask_question(payload: QueryInput):
    try:
        result = rag.answer(payload.query)
        return {"answer": result}
    except Exception as e:
        print(f"[ERROR] {e}")
        raise HTTPException(status_code=500, detail="internal server error")
