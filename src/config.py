import os

from dotenv import load_dotenv

load_dotenv()


def obter_chave_api(nome_variavel: str) -> str:
    chave = os.getenv(nome_variavel)
    if not chave:
        raise ValueError(f"{nome_variavel} não encontrada no arquivo .env")
    return chave
