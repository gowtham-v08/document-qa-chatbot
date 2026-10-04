"""
Document Q&A Chatbot (RAG)
Answers questions from uploaded text/PDF content using simple retrieval + optional LLM.
Works in demo mode without an API key (keyword retrieval + extractive answers).
"""

import streamlit as st
import re
from collections import Counter

st.set_page_config(
    page_title="Document Q&A Chatbot | RAG",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Document Q&A Chatbot (RAG)")
st.caption("Upload text from a PDF/document and ask questions. Uses chunking + retrieval (RAG pattern).")


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
        i += max(1, chunk_size - overlap)
    return chunks


def score_chunk(query: str, chunk: str) -> float:
    q = set(re.findall(r"[a-z0-9]+", query.lower()))
    c = Counter(re.findall(r"[a-z0-9]+", chunk.lower()))
    if not q:
        return 0.0
    hits = sum(c[w] for w in q)
    return hits / len(q)


def retrieve(query: str, chunks: list[str], top_k: int = 3) -> list[tuple[float, str]]:
    scored = [(score_chunk(query, ch), ch) for ch in chunks]
    scored.sort(key=lambda x: -x[0])
    return [(s, ch) for s, ch in scored[:top_k] if s > 0]


def answer_from_context(query: str, contexts: list[str]) -> str:
    if not contexts:
        return "I could not find relevant passages in the document for that question. Try rephrasing or uploading more content."
    # Extractive: return best passages joined
    joined = "\n\n---\n\n".join(contexts)
    return (
        f"Based on the most relevant parts of your document:\n\n{joined}\n\n"
        f"_(Demo mode: extractive RAG. Add an OpenAI/Groq API key later for generative answers.)_"
    )


with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top passages to retrieve", 1, 5, 3)
    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown(
        "1. Split document into chunks  \n"
        "2. Score chunks vs your question  \n"
        "3. Return top passages as context (RAG)"
    )

uploaded = st.file_uploader("Upload a .txt file (or paste text below)", type=["txt"])
paste = st.text_area("Or paste document text here", height=180)

doc_text = ""
if uploaded is not None:
    doc_text = uploaded.read().decode("utf-8", errors="ignore")
elif paste.strip():
    doc_text = paste

if doc_text.strip():
    chunks = chunk_text(doc_text)
    st.success(f"Document loaded · {len(chunks)} chunks · {len(doc_text.split())} words")

    question = st.text_input("Ask a question about the document")
    if st.button("Get Answer", type="primary") and question.strip():
        hits = retrieve(question, chunks, top_k=top_k)
        contexts = [ch for _, ch in hits]
        st.subheader("Answer")
        st.write(answer_from_context(question, contexts))

        with st.expander("Retrieved passages (context)"):
            for i, (score, ch) in enumerate(hits, 1):
                st.markdown(f"**#{i} · relevance {score:.2f}**")
                st.write(ch)
else:
    st.info("Upload a .txt file or paste document text to start.")

st.markdown("---")
st.caption("Built with Python · Streamlit · RAG pattern  |  github.com/gowtham-v08/document-qa-chatbot")
