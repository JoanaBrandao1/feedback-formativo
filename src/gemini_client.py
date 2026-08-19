import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY não encontrada no arquivo .env")

client = genai.Client(api_key=api_key)

MODEL = "gemini-3.1-flash-lite"

def gerar_resposta(prompt: str) -> str:
    response = client.models.generate_content(model=MODEL, contents=prompt)
    return response.text