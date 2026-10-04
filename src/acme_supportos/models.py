from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

load_dotenv()

def model_id()-> str:
    model=os.getenv("MODEL")
    if not model:
        raise RuntimeError('Model name is not set in .env')
    return model

def get_model(**overrides):
    model_settings= {"max_retries": 2,
                    "timeout":20,
                    "max_tokens":30,
    }
    model_settings.update(overrides)
    return init_chat_model(model_id(), **model_settings)

from langchain_anthropic import ChatAnthropic
claude= ChatAnthropic(model="antropic:claude-sonnet-4-5", temperature=0)
