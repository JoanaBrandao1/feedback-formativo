"""
Coleta classificações dos modelos (Claude, Gemini, OpenAI) para as categorias
GP1-GP7 de todo o dataset e salva os resultados em data/processed/.

Se um resultado parcial já existir (de uma execução anterior ou interrompida),
os IDs já processados são pulados — só continua de onde parou.

Uso:
    python scripts/gerar_respostas.py            # dataset completo
    python scripts/gerar_respostas.py --n 10      # só as 10 primeiras linhas,
                                                   # salva em respostas_<modelo>_teste.csv
                                                   # (não mexe nos resultados da rodada completa)
"""

import argparse
import importlib
import sys
import time
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.append(str(RAIZ))

from src.modelos import MODELOS
from src.prompts import CATEGORIAS, montar_prompt, parse_resposta

CAMINHO_DATASET = RAIZ / "data" / "raw" / "dataset.csv"
PASTA_SAIDA = RAIZ / "data" / "processed"
TENTATIVAS_POR_LINHA = 3  # tentativas antes de desistir de uma linha (erros de rede/rate limit são passageiros)


def carregar_modelos() -> dict:
    """Importa o cliente de cada modelo separadamente, para permitir rodar só
    com as chaves de API que estiverem configuradas no .env."""
    modelos = {}
    for id_modelo, info in MODELOS.items():
        try:
            modelos[id_modelo] = importlib.import_module(f"src.{id_modelo}_client")
        except ValueError as erro:
            print(f"[aviso] {info['nome']} não será avaliado: {erro}")
    return modelos


def avaliar_modelo(nome_modelo: str, cliente, df: pd.DataFrame, caminho_saida: Path) -> None:
    ja_processados = set()
    if caminho_saida.exists():
        ja_processados = set(pd.read_csv(caminho_saida)["ID"])

    pendentes = df[~df["ID"].isin(ja_processados)]
    if ja_processados:
        print(f"  [{nome_modelo}] retomando: {len(ja_processados)} já processados, faltam {len(pendentes)}")
    if pendentes.empty:
        return

    for i, row in enumerate(pendentes.itertuples(), start=1):
        prompt = montar_prompt(row.Text)
        resposta = ""
        previsao = {categoria: None for categoria in CATEGORIAS}

        for tentativa in range(1, TENTATIVAS_POR_LINHA + 1):
            try:
                resposta = cliente.gerar_resposta(prompt)
                previsao = parse_resposta(resposta)
                break
            except Exception as erro:
                if tentativa < TENTATIVAS_POR_LINHA:
                    print(f"  [{nome_modelo}] erro no ID {row.ID} (tentativa {tentativa}/{TENTATIVAS_POR_LINHA}): {erro} — tentando de novo...")
                    time.sleep(5)
                else:
                    print(f"  [{nome_modelo}] erro no ID {row.ID}, desistindo após {TENTATIVAS_POR_LINHA} tentativas: {erro}")

        linha = pd.DataFrame([{"ID": row.ID, "resposta_bruta": resposta, **previsao}])
        linha.to_csv(caminho_saida, mode="a", header=not caminho_saida.exists(), index=False)

        print(f"  [{nome_modelo}] ID {row.ID} processado ({i}/{len(pendentes)})")
        time.sleep(1)  # evita estourar limite de requisições por segundo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n", type=int, default=None,
        help="Processa só as N primeiras linhas do dataset (teste rápido/barato); "
             "salva em respostas_<modelo>_teste.csv, sem tocar na rodada completa",
    )
    args = parser.parse_args()

    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CAMINHO_DATASET)

    sufixo = ""
    if args.n:
        df = df.head(args.n)
        sufixo = "_teste"

    modelos = carregar_modelos()
    if not modelos:
        print("Nenhuma chave de API configurada em .env. Configure ao menos uma e tente novamente.")
        return

    for nome_modelo, cliente in modelos.items():
        caminho_saida = PASTA_SAIDA / f"respostas_{nome_modelo}{sufixo}.csv"
        if sufixo and caminho_saida.exists():
            caminho_saida.unlink()  # teste sempre roda do zero, pra refletir o prompt atual

        descricao = f"{len(df)} feedbacks de teste" if sufixo else f"{len(df)} feedbacks (dataset completo)"
        print(f"\nAvaliando {nome_modelo} ({descricao})...")
        avaliar_modelo(nome_modelo, cliente, df, caminho_saida)
        print(f"  -> resultados em {caminho_saida}")


if __name__ == "__main__":
    main()
