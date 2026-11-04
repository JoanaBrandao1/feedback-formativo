from openai import OpenAI

from src.config import obter_chave_api

client = OpenAI(api_key=obter_chave_api("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"


def gerar_resposta(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
