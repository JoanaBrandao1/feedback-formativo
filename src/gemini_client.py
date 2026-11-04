from google import genai
from google.genai import types

from src.config import obter_chave_api

client = genai.Client(api_key=obter_chave_api("GEMINI_API_KEY"))

MODEL = "gemini-3.1-flash-lite"

CONFIG = types.GenerateContentConfig(
    temperature=0,
    response_mime_type="application/json",
)


def gerar_resposta(prompt: str) -> str:
    response = client.models.generate_content(model=MODEL, contents=prompt, config=CONFIG)
    return response.text
