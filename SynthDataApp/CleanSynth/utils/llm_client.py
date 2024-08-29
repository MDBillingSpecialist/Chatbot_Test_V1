import openai
import os

def setup_llm_client():
    return openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
