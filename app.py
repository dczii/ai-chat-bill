from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from typing import cast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate

import chainlit as cl
import os
import tiktoken

def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-4o").encode(
        text,
    )
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function = tiktoken_len
)


@cl.on_chat_start
async def on_chat_start():
    elements = [
        cl.Image(name="logo", display="inline",
                 url="https://w7.pngwing.com/pngs/630/502/png-transparent-artificial-intelligence-computer-icons-robotics-artificial-intelligence-icon-angle-text-computer-thumbnail.png")
    ]
    
    await cl.Message(content="Hello! Welcome to Danilo's Chatbot!",
                     elements=elements).send()

    cl.user_session.set("chat_history", [])
    await cl.Message(content="Indexing AI bills").send()

    # LOAD PDF
    directory_loader = DirectoryLoader("bills", glob="**/*.pdf", loader_cls=PyMuPDFLoader)

    bill_knowledge_resources = directory_loader.load()

    documents = text_splitter.transform_documents(bill_knowledge_resources)
    store = LocalFileStore("./cache/")

    # Stronger system prompt: assume ambiguous = AI bills
    system_prompt = (
        "You are an expert lawyer specializing in AI bills. "
        "You must always interpret ambiguous or follow-up questions as referring to AI-related bills "
        "and continue the same topic unless the user explicitly asks for a different subject. "
        "Answer strictly using the provided context from AI-related bills. "
        "If the required information is not in the context, say: "
        "'I don't know. I can only answer questions related to AI bills based on the provided documents.' "
        "Cite specific sections if possible."
    )

    core_embeddings_model = OpenAIEmbeddings()
    embedder = CacheBackedEmbeddings.from_bytes_store(
        core_embeddings_model, store, namespace=core_embeddings_model.model
    )
    # make async docsearch
    docsearch = await cl.make_async(FAISS.from_documents)(documents, embedder)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human",
             "Use the following context to answer the user's question.\n\n"
             "Context:\n{context}\n\n"
             "Question: {question}")
        ]
    )

    llm = ChatOpenAI(temperature=0, streaming=True)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt,
            "document_variable_name": "context"  # must match the {context} in the prompt
        },
    )

    if chain is None:
        await cl.Message(content="Failed to build retriever from PDFs in ./bills.").send()
        return

    # A small condenser that rewrites any user turn into a standalone AI-bill question
    rewriter_prompt = ChatPromptTemplate.from_template(
        "You rewrite user questions into a single, clear standalone query about AI bills.\n"
        "Conversation so far:\n{history}\n\n"
        "User question: {question}\n\n"
        "Rewrite the question so it clearly refers to AI bills (assume AI-bill context if ambiguous). "
        "Do not answer; only rewrite the question."
    )
    question_rewriter = rewriter_prompt | llm | StrOutputParser()

    cl.user_session.set("question_rewriter", question_rewriter)
    cl.user_session.set("chain", chain)

    await cl.Message(content="Index built!").send()


@cl.on_message
async def on_message(message: cl.Message):
    await cl.Message(content="Thinking...").send()
    chain = cl.user_session.get("chain")
    if chain is None:
        await cl.Message(content="Search chain not ready yet. Please try again.").send()
        return

    # Let LangChain/LLM stream via Chainlit’s callback; RetrievalQA returns a dict, so we don't stream tokens ourselves.
    cb = cl.LangchainCallbackHandler()  # streams model tokens to the UI

    result = await chain.ainvoke({"query": message.content}, config=RunnableConfig(callbacks=[cb]))

    # Extract the answer text robustly
    answer = result.get("result") or result.get("answer") or ""
    sources = result.get("source_documents") or []

    # Build a short sources footer
    answer = result.get("result") or result.get("answer") or ""
    sources = result.get("source_documents") or []
    if sources:
        lines = []
        for i, d in enumerate(sources, 1):
            meta = d.metadata or {}
            title = meta.get("source") or meta.get("file_path") or f"Document {i}"
            lines.append(f"{i}. {title}")
        answer += "\n\n**Sources:**\n" + "\n".join(lines)

    await cl.Message(content=answer).send()

from chainlit.server import app  # ASGI for Vercel

if __name__ == "__main__":
    import chainlit as cl, os
    cl.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
