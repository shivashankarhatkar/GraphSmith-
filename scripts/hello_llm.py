import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

model= init_chat_model(os.environ["MODEL"], temperature= 0)
reply = model.invoke("Hello! I am Shiv, Learning LangChain and LangGraph frameworks in depth")
print(reply.text)
print(reply.usage_metadata)
