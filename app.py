from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from typing import cast
import os
import chainlit as cl

@cl.on_chat_start
async def on_chat_start():
    model = ChatOpenAI(streaming=True)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a helpful design thinking coach. Greet the user with, Hello there! Give me a great idea, let us see what we can do about it. Based on the user's input, throw random ideas that range from cosine similarity of 0.20 to -0.5",
            ),
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

# For Vercel deployment - expose the ASGI app
from chainlit.server import app

# This is what Vercel will use
def handler(request):
    return app(request)

# For local development
if __name__ == "__main__":
    cl.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))