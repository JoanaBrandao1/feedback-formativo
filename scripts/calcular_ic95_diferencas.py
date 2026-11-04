"""
Intervalo de confiança de 95% da DIFERENÇA entre cada par de modelos
(bootstrap pareado, 1000 reamostragens, semente fixa), para F1, MCC e Kappa,
em cada categoria com positivos no gabarito e na média dessas categorias.

Responde à pergunta "um modelo é melhor que o outro?": se o intervalo da
diferença contém 0, não há evidência de diferença entre os dois.

Não chama nenhuma API e não altera nenhum arquivo existente: escreve só
data/processed/diferencas_ic95.csv.

Uso:
    python scripts/calcular_ic95_diferencas.py
"""

import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.append(str(RAIZ))

from src.intervalos_confianca import N_BOOTSTRAP, SEMENTE, diferenca_pareada
from src.modelos import MODELOS
from src.prompts import CATEGORIAS

CAMINHO_DATASET = RAIZ / "data" / "raw" / "dataset.csv"
PASTA_PROCESSADOS = RAIZ / "data" / "processed"
CAMINHO_SAIDA = PASTA_PROCESSADOS / "diferencas_ic95.csv"


def main():
    df_gabarito = pd.read_csv(CAMINHO_DATASET)
    # GP7 não tem positivos no gabarito (métricas N/A), então fica de fora
    categorias = [c for c in CATEGORIAS if df_gabarito[c].sum() > 0]

    respostas = {}
    for id_modelo, info in MODELOS.items():
        caminho = PASTA_PROCESSADOS / f"respostas_{id_modelo}.csv"
        if caminho.exists():
            respostas[info["nome"]] = pd.read_csv(caminho)
        else:
            print(f"[aviso] {caminho.name} não encontrado, {info['nome']} ficou de fora.")

    if len(respostas) < 2:
        print("São necessários pelo menos dois modelos com respostas para comparar.")
        return

    resultados = []
    for nome_a, nome_b in combinations(respostas, 2):
        print(f"Calculando {nome_a} − {nome_b} ({N_BOOTSTRAP} reamostragens pareadas, semente {SEMENTE})...")
        df = diferenca_pareada(df_gabarito, respostas[nome_a], respostas[nome_b], categorias)
        df.insert(0, "modelo_b", nome_b)
        df.insert(0, "modelo_a", nome_a)
        resultados.append(df)

    df_final = pd.concat(resultados, ignore_index=True)
    df_final.to_csv(CAMINHO_SAIDA, index=False)
    print(f"  -> {len(df_final)} linhas salvas em {CAMINHO_SAIDA}\n")

    media = df_final[df_final["categoria"] == "Média"]
    for _, linha in media.iterrows():
        conclusao = "diferença" if linha["ic_exclui_zero"] else "sem evidência de diferença"
        print(
            f"{linha['modelo_a']} − {linha['modelo_b']} | {linha['metrica'].upper():5} média: "
            f"{linha['diferenca']:+.3f} [{linha['ic95_inferior']:+.3f}; {linha['ic95_superior']:+.3f}] → {conclusao}"
        )


if __name__ == "__main__":
    main()
