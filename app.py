from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from typing import cast
import os
import chainlit as cl

SYSTEM_PROMPT = (
    "You are an expert lawyer specializing in AI bills in the Philippines. "
    "Always interpret ambiguous or follow-up questions as referring to AI-related bills and continue the same topic "
    "unless the user explicitly switches subjects. "
    "Answer strictly using the provided context from AI-related bills. "
    "If the required information is not in the context, reply:\n"
    "'I don't know. I can only answer questions related to AI bills based on the provided documents.' "
    "Cite specific sections if possible."
)

@cl.on_chat_start
async def on_chat_start():
    elements = [
        cl.Image(
            name="logo",
            display="inline",
            path="./static/Logo.png",
        )
    ]
    await cl.Message(content="Hello! Welcome to Danilo's Chatbot!", elements=elements).send()

    cl.user_session.set("chat_history", [])

    model = ChatOpenAI(streaming=True)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)

@cl.on_message
async def on_message(message: cl.Message):
    runnable = cast(Runnable, cl.user_session.get("runnable"))
    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()

# -----------------------------
# Railway / ASGI entrypoint
# -----------------------------
# Ensure Chainlit knows what app to load when served as ASGI
os.environ.setdefault("CHAINLIT_APP", __file__)
os.environ.setdefault("CHAINLIT_HOST", "0.0.0.0")
os.environ.setdefault("CHAINLIT_PORT", os.environ.get("PORT", "8000"))

# Expose the ASGI app for uvicorn/gunicorn
from chainlit.server import app as chainlit_asgi_app
app = chainlit_asgi_app  # <- Railway will serve this via uvicorn

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ["CHAINLIT_HOST"], port=int(os.environ["CHAINLIT_PORT"]))
