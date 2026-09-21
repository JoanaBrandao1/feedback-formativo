import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from src.prompts import CATEGORIAS


def calcular_metricas(df_previsto: pd.DataFrame, df_verdadeiro: pd.DataFrame) -> pd.DataFrame:
    """Compara as previsões de um modelo com o gabarito humano do dataset,
    categoria por categoria (GP1-GP7).

    df_previsto: colunas ID, GP1..GP7 (0/1/None) — saída de scripts/gerar_respostas.py
    df_verdadeiro: colunas ID, GP1..GP7 (0/1) — o dataset original (gabarito humano)
    """
    comparacao = df_verdadeiro[["ID", *CATEGORIAS]].merge(
        df_previsto[["ID", *CATEGORIAS]], on="ID", suffixes=("_real", "_previsto")
    )

    linhas = []
    for categoria in CATEGORIAS:
        col_real = f"{categoria}_real"
        col_previsto = f"{categoria}_previsto"
        validos = comparacao.dropna(subset=[col_previsto])
        if validos.empty:
            continue

        y_real = validos[col_real].astype(int)
        y_previsto = validos[col_previsto].astype(int)

        # MCC e Kappa ficam indefinidos (0/0) quando real e previsto são
        # constantes (ex.: categoria sem nenhum "1" e modelo também só prevê
        # "0"); nesse caso não há o que medir, então tratamos como 0.
        with np.errstate(invalid="ignore", divide="ignore"):
            mcc = matthews_corrcoef(y_real, y_previsto)
            kappa = cohen_kappa_score(y_real, y_previsto)

        linhas.append(
            {
                "categoria": categoria,
                "acuracia": accuracy_score(y_real, y_previsto),
                "precisao": precision_score(y_real, y_previsto, zero_division=0),
                "recall": recall_score(y_real, y_previsto, zero_division=0),
                "f1": f1_score(y_real, y_previsto, zero_division=0),
                "mcc": 0.0 if np.isnan(mcc) else mcc,
                "kappa": 0.0 if np.isnan(kappa) else kappa,
                "n_amostras": len(validos),
            }
        )

    return pd.DataFrame(linhas)
