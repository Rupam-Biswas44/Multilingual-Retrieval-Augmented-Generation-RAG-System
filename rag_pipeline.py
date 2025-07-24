from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.docstore.in_memory import InMemoryDocstore

import faiss
import numpy as np
class SimpleRAG:
    def __init__(self, pdf_path: str):
        print("[INFO] Loading and splitting PDF...")
        self.chunks = self.load_and_split_pdf(pdf_path)
        print(f"[INFO] {len(self.chunks)} chunks loaded.")
        self.texts = [chunk.page_content for chunk in self.chunks]
        self.metadatas = [chunk.metadata for chunk in self.chunks]

        self.embedding_model = OpenAIEmbeddings(
            openai_api_key="my api key............", 
            model="text-embedding-3-small"
        )
        vector_list = self.embedding_model.embed_documents(self.texts)
        vector_array = np.array(vector_list).astype("float32")

        dim = vector_array.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(vector_array)

        # Docstore
        index_to_docstore_id = {i: str(i) for i in range(len(self.texts))}
        docstore = InMemoryDocstore({
            str(i): Document(page_content=self.texts[i], metadata=self.metadatas[i])
            for i in range(len(self.texts))
        })

        self.vectorstore = FAISS(
            embedding_function=self.embedding_model,
            index=self.index,
            index_to_docstore_id=index_to_docstore_id,
            docstore=docstore
        )
        self.llm = ChatOpenAI(
            temperature=0.2,
            model_name="gpt-3.5-turbo",
            openai_api_key="...my api key....."  
        )

        self.chat_history = []  

    def load_and_split_pdf(self, pdf_path: str):
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=30,
            separators=["\n\n", "\n", "।", ".", "!", "?", ":", "-", "–"]
        )
        return splitter.split_documents(docs)

    def format_chat_history(self, max_turns=3):
        history = self.chat_history[-max_turns:]
        return "\n".join([f"প্রশ্ন: {h['question']}\nউত্তর: {h['answer']}" for h in history])

    def answer(self, query: str) -> str:
        print(f"[INFO] Query: {query}")

        docs = self.vectorstore.max_marginal_relevance_search(query, k=10, fetch_k=20)
        docs = [doc for doc in docs if len(doc.page_content.strip()) > 20]

        context = "\n".join([doc.page_content for doc in docs])
        history = self.format_chat_history()

        print("\n[DEBUG] Retrieved Context:")
        print(context)
        print("=" * 60)

        
        prompt = f"""
        তোমার কাজ হলো শুধু নিচের প্রসঙ্গ এবং পূর্ববর্তী প্রশ্নোত্তর ব্যবহার করে প্রশ্নের নির্ভুল উত্তর দেয়া। কল্পনা করে উত্তর দেবে না।
        যদি প্রসঙ্গ বা পূর্ববর্তী আলোচনায় তথ্য না থাকে, বলো "প্রসঙ্গ থেকে উত্তর খুঁজে পাওয়া যায়নি।"

        প্রসঙ্গ:
        \"\"\"{context}\"\"\"

        পূর্ববর্তী প্রশ্নোত্তর:
        {history}

        বর্তমান প্রশ্ন:
        {query}

        উত্তর বাংলায় এক লাইনে দাও (শুধু প্রাসঙ্গিক তথ্য ব্যবহার করো):
        """

        response = self.llm.invoke(prompt)
        answer = response.content.strip()
        self.chat_history.append({"question": query, "answer": answer})
        return answer