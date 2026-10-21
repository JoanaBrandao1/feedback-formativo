"""
Calcula o intervalo de confiança de 95% (bootstrap, 1000 reamostragens, semente
fixa) de F1, MCC e Kappa de cada modelo em cada categoria, a partir dos CSVs
que JÁ existem em data/processed/ (respostas_<modelo>.csv).

Não chama nenhuma API e não altera nenhum arquivo existente: escreve só
data/processed/metricas_ic95.csv (que o app lê para mostrar os intervalos).

Uso:
    python scripts/calcular_ic95.py
"""

import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.append(str(RAIZ))

from src.intervalos_confianca import N_BOOTSTRAP, SEMENTE, calcular_ic95
from src.modelos import MODELOS

CAMINHO_DATASET = RAIZ / "data" / "raw" / "dataset.csv"
PASTA_PROCESSADOS = RAIZ / "data" / "processed"
CAMINHO_SAIDA = PASTA_PROCESSADOS / "metricas_ic95.csv"


def main():
    df_gabarito = pd.read_csv(CAMINHO_DATASET)

    resultados = []
    for id_modelo, info in MODELOS.items():
        caminho = PASTA_PROCESSADOS / f"respostas_{id_modelo}.csv"
        if not caminho.exists():
            print(f"[aviso] {caminho.name} não encontrado, {info['nome']} ficou de fora.")
            continue
        print(f"Calculando IC95% de {info['nome']} ({N_BOOTSTRAP} reamostragens, semente {SEMENTE})...")
        resultados.append(calcular_ic95(pd.read_csv(caminho), df_gabarito, info["nome"]))

    if not resultados:
        print("Nenhum respostas_<modelo>.csv encontrado em data/processed/.")
        return

    df_ic = pd.concat(resultados, ignore_index=True)
    df_ic.to_csv(CAMINHO_SAIDA, index=False)
    print(f"  -> {len(df_ic)} linhas salvas em {CAMINHO_SAIDA}")


if __name__ == "__main__":
    main()
