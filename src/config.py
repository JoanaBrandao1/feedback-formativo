import os

from dotenv import load_dotenv

load_dotenv()


def obter_chave_api(nome_variavel: str) -> str:
    """Lê uma chave de API do .env, ou lança erro se ela não estiver configurada."""
    chave = os.getenv(nome_variavel)
    if not chave:
        raise ValueError(f"{nome_variavel} não encontrada no arquivo .env")
    return chave
