from anthropic import Anthropic

from src.config import obter_chave_api

client = Anthropic(api_key=obter_chave_api("CLAUDE_API_KEY"))

MODEL = "claude-sonnet-5"


def gerar_resposta(prompt: str) -> str:
    # Nota: `temperature` é rejeitado pela API para o claude-sonnet-5 ("deprecated for
    # this model") — o modelo não aceita mais esse controle manual de aleatoriedade.
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return next(block.text for block in response.content if block.type == "text")
