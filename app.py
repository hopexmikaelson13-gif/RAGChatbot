from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st
from streamlit.runtime import exists as streamlit_runtime_exists
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PDF_DIR = Path(__file__).parent / "PDF Files"
SUPPORTED_SUFFIXES = {".pdf"}
RAG_SYSTEM_PROMPT = """You answer questions about a collection of PDF documents.
Use only the supplied context. Do not invent facts or use outside knowledge.
If the context does not answer the question, say: "I could not find that in the PDFs."
Cite every factual claim with the matching source marker, such as [1] or [2].
Prefer a concise answer, and distinguish conflicting findings when sources disagree.
"""


def _cache_data(function):
    if not streamlit_runtime_exists():
        return function
    return st.cache_data(show_spinner=False)(function)


def _cache_resource(function):
    if not streamlit_runtime_exists():
        return function
    return st.cache_resource(show_spinner=False)(function)


@_cache_data
def load_documents(pdf_dir: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for pdf_path in sorted(Path(pdf_dir).iterdir(), key=lambda path: path.name.lower()):
        if pdf_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        reader = PdfReader(str(pdf_path))
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").replace("\x00", " ").strip()
            if not text:
                continue
            words = text.split()
            chunk_size = 220
            overlap = 40
            for start in range(0, len(words), chunk_size - overlap):
                chunk = " ".join(words[start : start + chunk_size]).strip()
                if len(chunk) >= 80:
                    chunks.append(
                        {
                            "text": chunk,
                            "source": pdf_path.name,
                            "page": page_number,
                        }
                    )
    return chunks


@_cache_resource
def build_index(texts: tuple[str, ...]) -> tuple[TfidfVectorizer, Any]:
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_df=0.98)
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def retrieve(question: str, documents: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    vectorizer, matrix = build_index(tuple(document["text"] for document in documents))
    scores = cosine_similarity(vectorizer.transform([question]), matrix).ravel()
    ranked = scores.argsort()[::-1][:top_k]
    return [
        {**documents[index], "score": float(scores[index])}
        for index in ranked
        if scores[index] > 0
    ]


class LocalRAG:
    """Small programmatic adapter around the app's document and retrieval functions."""

    def __init__(self, chunk_size: int = 900, overlap: int = 150) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.documents: list[dict[str, Any]] = []

    def build(self) -> int:
        self.documents = load_documents(str(PDF_DIR))
        return len(self.documents)

    def search(self, question: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self.documents:
            self.build()
        return retrieve(question, self.documents, top_k)


def generate_answer(question: str, results: list[dict[str, Any]]) -> str:
    """Generate a local, dependency-free answer for tests and scripts."""
    return generated_answer(question, results, provider="Extractive")


def format_sources(context: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[{index}] {item['source']} (page {item['page']})\n{item['text']}"
        for index, item in enumerate(context, start=1)
    )


def build_rag_prompt(question: str, context: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build the grounded prompt sent to the answer model."""
    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Retrieved PDF context:\n{format_sources(context)}"
            ),
        },
    ]


def extractive_answer(context: list[dict[str, Any]]) -> str:
    if not context:
        return "I could not find relevant text in the PDFs. Try using terms that appear in the reports."
    excerpts = []
    for item in context[:3]:
        excerpt = item["text"]
        if len(excerpt) > 1400:
            excerpt = excerpt[:1400].rsplit(" ", 1)[0] + "..."
        excerpts.append(f"**{item['source']}**, page {item['page']}\n\n{excerpt}")
    return "I found these relevant passages:\n\n" + "\n\n---\n\n".join(excerpts)


def generated_answer(question: str, context: list[dict[str, Any]], provider: str) -> str:
    if provider == "Extractive" or not context:
        return extractive_answer(context)

    try:
        from openai import OpenAI

        if provider == "Ollama":
            client = OpenAI(
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                api_key="ollama",
            )
            model = os.getenv("OLLAMA_MODEL", "llama3.2")
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return "Set OPENAI_API_KEY or choose Extractive mode in the sidebar."
            client = OpenAI(api_key=api_key)
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        response = client.chat.completions.create(
            model=model,
            temperature=0.1,
            messages=build_rag_prompt(question, context),
        )
        return response.choices[0].message.content or extractive_answer(context)
    except Exception as error:
        return f"The {provider} model could not be reached ({error}).\n\n" + extractive_answer(context)


def main() -> None:
    st.set_page_config(page_title="PDF Research Chat", page_icon="?", layout="wide")
    st.title("PDF Research Chat")
    st.caption("Ask questions across the reports in this folder. Answers are grounded in retrieved PDF passages.")

    documents = load_documents(str(PDF_DIR))
    if not documents:
        st.error("No readable PDF text was found in this folder.")
        return

    with st.sidebar:
        st.subheader("Settings")
        provider = st.selectbox("Answer mode", ["Extractive", "OpenAI", "Ollama"])
        top_k = st.slider("Retrieved passages", min_value=2, max_value=8, value=4)
        st.divider()
        st.write(f"Indexed **{len(documents)}** passages")
        st.write(f"Found **{len({document['source'] for document in documents})}** PDFs")
        st.caption("OpenAI uses OPENAI_API_KEY. Ollama uses a local Ollama server.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask something about the PDFs...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching the PDFs..."):
                context = retrieve(question, documents, top_k)
                answer = generated_answer(question, context, provider)
                st.markdown(answer)
                if context:
                    with st.expander("Retrieved sources"):
                        for item in context:
                            st.caption(f"{item['source']} - page {item['page']} - score {item['score']:.3f}")
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()