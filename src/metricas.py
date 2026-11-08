import warnings

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

COLUNAS_METRICAS = ["acuracia", "precisao", "recall", "f1", "mcc", "kappa"]

METRICAS_SEM_POSITIVOS_INDEFINIDAS = ["precisao", "recall", "f1", "mcc", "kappa"]


def calcular_metricas(df_previsto: pd.DataFrame, df_verdadeiro: pd.DataFrame) -> pd.DataFrame:
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
                "n_positivos_gabarito": int(y_real.sum()),
                "n_positivos_previstos": int(y_previsto.sum()),
            }
        )

    return pd.DataFrame(linhas)


def marcar_nao_definidas(df_metricas: pd.DataFrame) -> pd.DataFrame:
    resultado = df_metricas.copy()
    sem_positivos = resultado["n_positivos_gabarito"] == 0
    resultado.loc[sem_positivos, METRICAS_SEM_POSITIVOS_INDEFINIDAS] = np.nan
    return resultado


def media_por_modelo(df_metricas: pd.DataFrame, ordem_modelos: list[str]) -> pd.DataFrame:
    base = df_metricas[df_metricas["n_positivos_gabarito"] > 0]
    return base.groupby("modelo")[COLUNAS_METRICAS].mean().reindex(ordem_modelos)


def resumir_falhas(df_previsto: pd.DataFrame) -> dict[str, int]:
    if "resposta_bruta" in df_previsto.columns:
        vazia = df_previsto["resposta_bruta"].fillna("").astype(str).str.strip() == ""
    else:
        vazia = pd.Series(False, index=df_previsto.index)
    return {
        "total": len(df_previsto),
        "resposta_vazia": int(vazia.sum()),
        "com_gp_nulo": int(df_previsto[CATEGORIAS].isna().any(axis=1).sum()),
    }


def comparar_execucoes(df_original: pd.DataFrame, df_nova: pd.DataFrame) -> pd.DataFrame:
    comparacao = df_original[["ID", *CATEGORIAS]].merge(
        df_nova[["ID", *CATEGORIAS]], on="ID", suffixes=("_original", "_nova")
    )

    linhas = []
    for categoria in CATEGORIAS:
        validos = comparacao.dropna(subset=[f"{categoria}_original", f"{categoria}_nova"])
        original = validos[f"{categoria}_original"].astype(int)
        nova = validos[f"{categoria}_nova"].astype(int)
        with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
            warnings.simplefilter("ignore")
            kappa = cohen_kappa_score(original, nova) if len(validos) else np.nan
        linhas.append(
            {
                "categoria": categoria,
                "n_comparados": len(validos),
                "pct_concordancia": float((original == nova).mean() * 100) if len(validos) else np.nan,
                "kappa": kappa,
                "positivos_original": int(original.sum()),
                "positivos_nova": int(nova.sum()),
            }
        )

    completos = comparacao.dropna()
    colunas_originais = [f"{c}_original" for c in CATEGORIAS]
    colunas_novas = [f"{c}_nova" for c in CATEGORIAS]
    identicos = (completos[colunas_originais].values == completos[colunas_novas].values).all(axis=1)
    linhas.append(
        {
            "categoria": "TODAS (7 GP idênticos)",
            "n_comparados": len(completos),
            "pct_concordancia": float(identicos.mean() * 100) if len(completos) else np.nan,
            "kappa": np.nan,
            "positivos_original": np.nan,
            "positivos_nova": np.nan,
        }
    )
    return pd.DataFrame(linhas)
