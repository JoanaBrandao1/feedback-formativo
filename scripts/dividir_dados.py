import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.append(str(RAIZ))

from src.prompts import CATEGORIAS

CAMINHO_DATASET = RAIZ / "data" / "raw" / "dataset.csv"
PASTA_SPLITS = RAIZ / "data" / "splits"
CAMINHO_DEV = PASTA_SPLITS / "dev_ids.csv"
CAMINHO_TESTE = PASTA_SPLITS / "teste_ids.csv"

N_DEV = 200
SEMENTE = 42


def dividir(df: pd.DataFrame, n_dev: int = N_DEV, semente: int = SEMENTE) -> tuple[list[int], list[int]]:
    frequencia = df[CATEGORIAS].sum()
    ordem_rara = [c for c in frequencia.sort_values().index if frequencia[c] > 0]

    def estrato(linha) -> str:
        for categoria in ordem_rara:
            if linha[categoria] == 1:
                return categoria
        return "nenhuma"

    estratos = df.apply(estrato, axis=1)
    rng = np.random.default_rng(semente)

    tamanhos = estratos.value_counts().sort_index()
    cota = tamanhos * n_dev / len(df)
    n_por_estrato = np.floor(cota).astype(int)
    faltam = n_dev - n_por_estrato.sum()
    for nome in (cota - n_por_estrato).sort_values(ascending=False).index[:faltam]:
        n_por_estrato[nome] += 1

    dev = []
    for nome, n in n_por_estrato.items():
        ids = df.loc[estratos == nome, "ID"].to_numpy()
        dev.extend(int(i) for i in rng.permutation(ids)[:n])
    dev = sorted(dev)
    teste = sorted(int(i) for i in df["ID"] if i not in set(dev))
    return dev, teste


def main():
    if CAMINHO_DEV.exists() or CAMINHO_TESTE.exists():
        print(f"A divisão já existe em {PASTA_SPLITS}. Nada foi alterado.")
        print("Ela deve ser feita uma única vez; só apague os arquivos se tiver certeza.")
        return

    df = pd.read_csv(CAMINHO_DATASET)
    dev, teste = dividir(df)

    PASTA_SPLITS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ID": dev}).to_csv(CAMINHO_DEV, index=False)
    pd.DataFrame({"ID": teste}).to_csv(CAMINHO_TESTE, index=False)

    resumo = pd.DataFrame(
        {
            "desenvolvimento": df[df["ID"].isin(dev)][CATEGORIAS].sum(),
            "teste": df[df["ID"].isin(teste)][CATEGORIAS].sum(),
            "total": df[CATEGORIAS].sum(),
        }
    )
    print(f"Desenvolvimento: {len(dev)} feedbacks -> {CAMINHO_DEV}")
    print(f"Teste: {len(teste)} feedbacks -> {CAMINHO_TESTE}\n")
    print("Positivos de cada boa prática em cada parte:")
    print(resumo.to_string())


if __name__ == "__main__":
    main()
