"""rag_chain.py - Local RAG chain with Qwen2.5 (Ollama) + Chroma + chat memory.

Setup
-----
    ollama pull qwen2.5
    ollama pull nomic-embed-text
    pip install -U langchain langchain-core langchain-community \
        langchain-ollama langchain-chroma langchain-text-splitters

Usage
-----
    python rag_chain.py --docs ./docs            # ingest (first run) and chat
    python rag_chain.py --docs ./docs --rebuild  # re-ingest from scratch
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

LLM_MODEL = "qwen2.5"
EMBED_MODEL = "nomic-embed-text"
PERSIST_DIR = "./chroma_db"
MAX_HISTORY_MESSAGES = 8  # keep memory small so the SLM context window is not exceeded

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer using ONLY the context below. "
    "If the answer is not in the context, say you don't know.\n\n"
    "Context:\n{context}"
)


def load_documents(docs_dir: str):
    """Load .txt and .md files from a directory (extend with PDF loaders as needed)."""
    docs = []
    for pattern in ("**/*.txt", "**/*.md"):
        loader = DirectoryLoader(
            docs_dir, glob=pattern, loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        docs.extend(loader.load())
    return docs


def build_vectorstore(docs_dir: str, rebuild: bool = False) -> Chroma:
    """Create the Chroma store on first run, otherwise reuse the persisted one."""
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    if rebuild and Path(PERSIST_DIR).exists():
        shutil.rmtree(PERSIST_DIR)

    if Path(PERSIST_DIR).exists():
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

    docs = load_documents(docs_dir)
    if not docs:
        raise SystemExit(f"No .txt or .md documents found in {docs_dir!r}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"Ingesting {len(docs)} documents as {len(chunks)} chunks...")
    return Chroma.from_documents(
        chunks, embeddings, persist_directory=PERSIST_DIR
    )


def format_docs(docs) -> str:
    return "\n\n".join(
        f"[{Path(d.metadata.get('source', 'unknown')).name}] {d.page_content}"
        for d in docs
    )


def build_chain(vectorstore: Chroma):
    """Retriever -> prompt (with history) -> Qwen2.5 -> string output."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]
    )

    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    sessions: dict[str, InMemoryChatMessageHistory] = {}

    def get_history(session_id: str) -> InMemoryChatMessageHistory:
        history = sessions.setdefault(session_id, InMemoryChatMessageHistory())
        history.messages = history.messages[-MAX_HISTORY_MESSAGES:]  # trim
        return history

    return RunnableWithMessageHistory(
        rag_chain,
        get_history,
        input_messages_key="question",
        history_messages_key="history",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with your docs via Qwen2.5")
    parser.add_argument("--docs", default="./docs", help="folder of .txt/.md files")
    parser.add_argument("--rebuild", action="store_true", help="re-ingest documents")
    args = parser.parse_args()

    chain = build_chain(build_vectorstore(args.docs, args.rebuild))
    config = {"configurable": {"session_id": "default"}}

    print("Ask a question (type 'exit' to quit).")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit", ""}:
            break
        print("Qwen:", chain.invoke({"question": question}, config=config))


if __name__ == "__main__":
    main()
