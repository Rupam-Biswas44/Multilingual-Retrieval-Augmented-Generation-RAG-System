import requests
import streamlit as st

st.set_page_config("Hi I am here For helping you with your exam MCQ preparation", layout="wide")

st.title("\U0001F4D8 Bangla HSC Book QA – OpenAI Powered By Rupam")
query = st.text_area("\u2753 Enter your question (বাংলা বা English):", height=100)

if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            res = requests.post("http://localhost:8000/query", json={"query": query})
            if res.ok:
                st.success("\u2705 Answer:")
                st.markdown(f"**{res.json()['answer']}**")
            else:
                st.error("Something went wrong!")