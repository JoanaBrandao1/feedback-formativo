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

# Sem nenhum "1" no gabarito, estas métricas não têm o que medir (a acurácia
# continua definida, mas só reflete o quanto o modelo marcou "0").
METRICAS_SEM_POSITIVOS_INDEFINIDAS = ["precisao", "recall", "f1", "mcc", "kappa"]


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
                "n_positivos_gabarito": int(y_real.sum()),
                "n_positivos_previstos": int(y_previsto.sum()),
            }
        )

    return pd.DataFrame(linhas)


def marcar_nao_definidas(df_metricas: pd.DataFrame) -> pd.DataFrame:
    """Devolve uma cópia em que precisão, recall, F1, MCC e Kappa viram NaN
    (= "N/A") nas categorias sem nenhum positivo no gabarito.

    `calcular_metricas` continua devolvendo 0 nesses casos (convenção do
    sklearn); esta função é só para a apresentação, para não confundir
    "não dá para medir" com "o modelo errou tudo".
    """
    resultado = df_metricas.copy()
    sem_positivos = resultado["n_positivos_gabarito"] == 0
    resultado.loc[sem_positivos, METRICAS_SEM_POSITIVOS_INDEFINIDAS] = np.nan
    return resultado


def media_por_modelo(
    df_metricas: pd.DataFrame, ordem_modelos: list[str], somente_com_positivos: bool = True
) -> pd.DataFrame:
    """Média simples (sem pesos) das métricas por modelo.

    Com somente_com_positivos=True, só entram as categorias que têm ao menos um
    "1" no gabarito. Com False, entram todas (a média antiga, "com GP7").
    """
    base = df_metricas
    if somente_com_positivos:
        base = df_metricas[df_metricas["n_positivos_gabarito"] > 0]
    return base.groupby("modelo")[COLUNAS_METRICAS].mean().reindex(ordem_modelos)


def resumir_falhas(df_previsto: pd.DataFrame) -> dict[str, int]:
    """Conta respostas que falharam ou ficaram incompletas em um CSV de respostas.

    - resposta_vazia: a API falhou nas tentativas e nada foi salvo;
    - com_gp_nulo: pelo menos um GP ficou sem valor 0/1 (inclui as vazias).
    """
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
    """Mede o quanto duas execuções do MESMO modelo concordam entre si, por GP.

    Devolve uma linha por categoria (concordância em % e Kappa de Cohen) mais
    uma linha "TODAS" com o % de feedbacks em que os 7 GP saíram idênticos.
    O Kappa fica NaN quando é indefinido (ex.: as duas execuções só marcaram "0").
    """
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
