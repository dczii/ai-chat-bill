# app.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List

import chainlit as cl

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader

from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

import tiktoken

# ----------------------------
# Config: tune these for size/latency
# ----------------------------
EMBED_MODEL = "text-embedding-3-small"  # light+fast, good quality
CHUNK_SIZE = 1800                        # ↑ larger chunk -> fewer chunks -> smaller index
CHUNK_OVERLAP = 120                      # modest overlap
TOP_K = 4                                # retriever neighbors
BILLS_DIR = Path(__file__).parent / "bills"
INDEX_DIR = Path(__file__).parent / "index"   # prebuilt FAISS index recommended on Vercel

# ----------------------------
# Token length fn (tiktoken-based)
# ----------------------------
def tiktoken_len(text: str) -> int:
    # Use a recent chat model's tokenizer as a proxy; adjust if you prefer
    enc = tiktoken.encoding_for_model("gpt-4o")
    return len(enc.encode(text))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=tiktoken_len,
)

# ----------------------------
# Lazy singletons (kept in process memory)
# ----------------------------
_vectorstore: Optional[FAISS] = None
_retriever = None

def build_embeddings_cache() -> CacheBackedEmbeddings:
    """
    Wrap OpenAIEmbeddings with a small on-disk cache to avoid recomputing
    the same chunks across cold starts.
    """
    core = OpenAIEmbeddings(model=EMBED_MODEL)
    store = LocalFileStore(str(Path(__file__).parent / "cache"))
    return CacheBackedEmbeddings.from_bytes_store(core, store, namespace=core.model)

def load_or_build_vectorstore() -> FAISS:
    """
    Prefer loading a prebuilt FAISS index from ./index.
    If missing, build from PDFs in ./bills exactly once per process.
    """
    global _vectorstore

    if _vectorstore is not None:
        return _vectorstore

    embedder = build_embeddings_cache()

    if INDEX_DIR.exists():
        # Fast path: load prebuilt index
        _vectorstore = FAISS.load_local(
            str(INDEX_DIR),
            embedder,
            allow_dangerous_deserialization=True,  # required by LC for FAISS load
        )
        return _vectorstore

    # Slow path: build from PDFs (avoid this on Vercel; prebuild instead)
    if not BILLS_DIR.exists():
        raise FileNotFoundError(
            f"Neither index/ nor bills/ exist. "
            f"Commit a prebuilt FAISS index under {INDEX_DIR} or add PDFs under {BILLS_DIR}."
        )

    loader = DirectoryLoader(str(BILLS_DIR), glob="**/*.pdf", loader_cls=PyMuPDFLoader)
    docs = loader.load()
    chunks = text_splitter.split_documents(docs)

    _vectorstore = FAISS.from_documents(chunks, embedder)

    # Optional: save to disk so future cold starts on same build are faster (if writable)
    try:
        INDEX_DIR.mkdir(exist_ok=True)
        _vectorstore.save_local(str(INDEX_DIR))
    except Exception:
        # On Vercel the FS is read-only at runtime—ignore.
        pass

    return _vectorstore

def get_retriever():
    global _retriever
    if _retriever is not None:
        return _retriever
    vs = load_or_build_vectorstore()
    _retriever = vs.as_retriever(search_kwargs={"k": TOP_K})
    return _retriever

# ----------------------------
# System & helper prompts
# ----------------------------
SYSTEM_PROMPT = (
    "You are an expert lawyer specializing in AI bills. "
    "Always interpret ambiguous or follow-up questions as referring to AI-related bills and continue the same topic "
    "unless the user explicitly switches subjects. "
    "Answer strictly using the provided context from AI-related bills. "
    "If the required information is not in the context, reply:\n"
    "'I don't know. I can only answer questions related to AI bills based on the provided documents.' "
    "Cite specific sections if possible."
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human",
         "Use the following context to answer the user's question.\n\n"
         "Context:\n{context}\n\n"
         "Question: {question}")
    ]
)

REWRITER_PROMPT = ChatPromptTemplate.from_template(
    "Rewrite the user's question as a single, clear standalone query about AI bills.\n"
    "Conversation so far:\n{history}\n\n"
    "User question: {question}\n\n"
    "Return only the rewritten question."
)

# ----------------------------
# Chainlit events
# ----------------------------
@cl.on_chat_start
async def on_chat_start():
    # small header image
    elements = [
        cl.Image(
            name="logo",
            display="inline",
            url="https://w7.pngwing.com/pngs/630/502/png-transparent-artificial-intelligence-computer-icons-robotics-artificial-intelligence-icon-angle-text-computer-thumbnail.png",
        )
    ]
    await cl.Message(content="Hello! Welcome to Danilo's Chatbot!", elements=elements).send()

    cl.user_session.set("chat_history", [])
    await cl.Message(content="Preparing knowledge base…").send()

    # Heavy init (index load/build) — do it once, off the event loop
    try:
        loader_fn = cl.make_async(load_or_build_vectorstore)
        await loader_fn()
    except Exception as e:
        await cl.Message(
            content=f"Failed to initialize vector store: {e}"
        ).send()
        return

    # LLM (streaming on) — temperature 0 for factual QA
    llm = ChatOpenAI(temperature=0, streaming=True)

    # Retrieval pipeline
    retriever = get_retriever()
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": ANSWER_PROMPT,
            "document_variable_name": "context",
        },
    )

    # Lightweight question rewriter (kept if you want to use later)
    question_rewriter = REWRITER_PROMPT | llm | StrOutputParser()

    cl.user_session.set("chain", chain)
    cl.user_session.set("question_rewriter", question_rewriter)

    await cl.Message(content="Knowledge base is ready! Ask me about AI bills.").send()


@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get("chain")
    if chain is None:
        await cl.Message(content="Search chain not ready yet. Please try again.").send()
        return

    cb = cl.LangchainCallbackHandler()  # streams tokens to Chainlit UI

    # If you want to enforce rewriting, uncomment:
    # qr = cl.user_session.get("question_rewriter")
    # rewritten = await qr.ainvoke({"history": cl.user_session.get("chat_history", []), "question": message.content})
    # user_query = rewritten
    user_query = message.content

    result = await chain.ainvoke({"query": user_query}, config={"callbacks": [cb]})

    answer = result.get("result") or result.get("answer") or ""
    sources = result.get("source_documents") or []

    if sources:
        lines: List[str] = []
        for i, d in enumerate(sources, 1):
            meta = d.metadata or {}
            title = (
                meta.get("source")
                or meta.get("file_path")
                or meta.get("filename")
                or f"Document {i}"
            )
            lines.append(f"{i}. {title}")
        answer += "\n\n**Sources:**\n" + "\n".join(lines)

    await cl.Message(content=answer).send()

# ----------------------------
# ASGI export for Vercel
# ----------------------------
from chainlit.server import app  # ASGI app

# Optional: local dev entry
if __name__ == "__main__":
    cl.run(host="0.0.0.0", port=int(os.environ.get("PORT", 3000)))
