import os

from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

api_key = os.getenv("CLAUDE_API_KEY")

if not api_key:
    raise ValueError("CLAUDE_API_KEY não encontrada no arquivo .env")

client = Anthropic(api_key=api_key)

MODEL = "claude-sonnet-5"


def gerar_resposta(prompt: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return next(block.text for block in response.content if block.type == "text")
