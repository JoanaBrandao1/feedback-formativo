import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.append(str(RAIZ))

from src.intervalos_confianca import N_BOOTSTRAP, SEMENTE, calcular_ic95, calcular_ic95_diferencas
from src.modelos import MODELOS

CAMINHO_DATASET = RAIZ / "data" / "raw" / "dataset.csv"
PASTA_PROCESSADOS = RAIZ / "data" / "processed"
CAMINHO_SAIDA = PASTA_PROCESSADOS / "metricas_ic95.csv"
CAMINHO_SAIDA_DIFERENCAS = PASTA_PROCESSADOS / "diferencas_ic95.csv"


def main():
    df_gabarito = pd.read_csv(CAMINHO_DATASET)

    previsoes = {}
    for id_modelo, info in MODELOS.items():
        caminho = PASTA_PROCESSADOS / f"respostas_{id_modelo}.csv"
        if not caminho.exists():
            print(f"[aviso] {caminho.name} não encontrado, {info['nome']} ficou de fora.")
            continue
        previsoes[info["nome"]] = pd.read_csv(caminho)

    if not previsoes:
        print("Nenhum respostas_<modelo>.csv encontrado em data/processed/.")
        return

    resultados = []
    for nome, df_previsto in previsoes.items():
        print(f"Calculando IC95% de {nome} ({N_BOOTSTRAP} reamostragens, semente {SEMENTE})...")
        resultados.append(calcular_ic95(df_previsto, df_gabarito, nome))

    df_ic = pd.concat(resultados, ignore_index=True)
    df_ic.to_csv(CAMINHO_SAIDA, index=False)
    print(f"  -> {len(df_ic)} linhas salvas em {CAMINHO_SAIDA}")

    if len(previsoes) < 2:
        print("Só um modelo disponível: não há diferenças entre modelos para calcular.")
        return

    print(f"Calculando IC95% das diferenças entre pares de modelos ({N_BOOTSTRAP} reamostragens pareadas)...")
    df_diferencas = calcular_ic95_diferencas(previsoes, df_gabarito)
    df_diferencas.to_csv(CAMINHO_SAIDA_DIFERENCAS, index=False)
    print(f"  -> {len(df_diferencas)} linhas salvas em {CAMINHO_SAIDA_DIFERENCAS}")


if __name__ == "__main__":
    main()
