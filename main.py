import os
from dotenv import load_dotenv
from app.api import app
import uvicorn
from app.utils import load_and_split_pdf

chunks = load_and_split_pdf("data/HSC26-Bangla1st-Paper.pdf")

for i, chunk in enumerate(chunks):
    if "সুপুরুষ" in chunk.page_content or "অনুপম" in chunk.page_content:
        print(f"Chunk {i}:\n{chunk.page_content}\n{'='*50}")

if __name__ == "__main__":
    load_dotenv()
    import uvicorn
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
