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

from llm_api import openai_chatbot_chain
import chainlit as cl

#|--------------------------------------------------------------------------|
#|                            On Boarding                                   |
#|--------------------------------------------------------------------------|
@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": SYSTEM_PROMPT}],
    )
    app_user = cl.user_session.get("user")
    await cl.Message(f"Hello User").send()

#|--------------------------------------------------------------------------|
#|                               Chat                                       |
#|--------------------------------------------------------------------------|
@cl.on_message
async def main(user_input: cl.Message):
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": user_input.content})

    llm_output = cl.Message(content="")
    await llm_output.send()

    stream = await openai_chatbot_chain(message_history)

    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await llm_output.stream_token(token)

    message_history.append({"role": "assistant", "content": llm_output.content})
    await llm_output.update()