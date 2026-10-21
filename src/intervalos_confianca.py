"""Intervalos de confiança de 95% por bootstrap para F1, MCC e Kappa.

A ideia: o gabarito tem só 1000 feedbacks (e categorias como GP2 e GP4 têm
pouquíssimos positivos), então uma métrica calculada uma vez é uma estimativa
incerta. O bootstrap sorteia 1000 feedbacks COM reposição, recalcula a métrica
nessa "amostra alternativa", repete isso N_BOOTSTRAP vezes e usa os percentis
2,5% e 97,5% das métricas obtidas como intervalo (método do percentil).

Só numpy e scikit-learn: o sklearn calcula a estimativa pontual (a mesma da
tabela do app, via `calcular_metricas`); o numpy refaz F1/MCC/Kappa a partir das
contagens VP/FP/FN/VN de cada reamostragem, o que é bem mais rápido que chamar
o sklearn 1000 vezes por célula. As convenções para casos indefinidos são as
mesmas de `calcular_metricas` (0, em vez de erro).
"""

import numpy as np
import pandas as pd

from src.metricas import calcular_metricas
from src.prompts import CATEGORIAS

N_BOOTSTRAP = 1000
SEMENTE = 42
NIVEL_CONFIANCA = 95
METRICAS_COM_IC = ["f1", "mcc", "kappa"]


def _dividir(numerador: np.ndarray, denominador: np.ndarray) -> np.ndarray:
    """Divisão que devolve 0 onde o denominador é 0 (convenção usada no projeto)."""
    resultado = np.zeros_like(numerador, dtype=float)
    np.divide(numerador, denominador, out=resultado, where=denominador != 0)
    return resultado


def metricas_das_contagens(vp, fp, fn, vn) -> dict[str, np.ndarray]:
    """F1, MCC e Kappa a partir das contagens da matriz de confusão (vetores numpy).

    Mesmas fórmulas e mesmas convenções do sklearn/`calcular_metricas`:
    F1, MCC e Kappa indefinidos valem 0.
    """
    vp, fp, fn, vn = (np.asarray(x, dtype=float) for x in (vp, fp, fn, vn))
    total = vp + fp + fn + vn

    f1 = _dividir(2 * vp, 2 * vp + fp + fn)

    denominador_mcc = np.sqrt((vp + fp) * (vp + fn) * (vn + fp) * (vn + fn))
    mcc = _dividir(vp * vn - fp * fn, denominador_mcc)

    concordancia_observada = _dividir(vp + vn, total)
    concordancia_acaso = _dividir((vp + fp) * (vp + fn) + (vn + fn) * (vn + fp), total**2)
    kappa = _dividir(concordancia_observada - concordancia_acaso, 1 - concordancia_acaso)

    return {"f1": f1, "mcc": mcc, "kappa": kappa}


def bootstrap_metricas(
    y_real: np.ndarray, y_previsto: np.ndarray, rng: np.random.Generator, n_bootstrap: int = N_BOOTSTRAP
) -> dict[str, np.ndarray]:
    """Distribuição bootstrap (n_bootstrap valores) de F1, MCC e Kappa."""
    n = len(y_real)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    real = y_real[indices]
    previsto = y_previsto[indices]

    vp = (real & previsto).sum(axis=1)
    fp = ((1 - real) & previsto).sum(axis=1)
    fn = (real & (1 - previsto)).sum(axis=1)
    vn = n - vp - fp - fn
    return metricas_das_contagens(vp, fp, fn, vn)


def calcular_ic95(
    df_previsto: pd.DataFrame,
    df_verdadeiro: pd.DataFrame,
    nome_modelo: str,
    n_bootstrap: int = N_BOOTSTRAP,
    semente: int = SEMENTE,
) -> pd.DataFrame:
    """IC95% de F1, MCC e Kappa de um modelo, em todas as categorias.

    Devolve formato "longo": uma linha por (categoria, métrica). Em categorias
    sem nenhum positivo no gabarito (ex.: GP7) a métrica não está definida, então
    estimativa e IC ficam vazios (NaN) com uma observação explicando.

    A semente depende só da categoria: modelos com as mesmas linhas válidas
    usam exatamente as mesmas reamostragens (bootstrap pareado), o que deixa a
    comparação entre modelos mais justa.
    """
    pontuais = calcular_metricas(df_previsto, df_verdadeiro).set_index("categoria")
    comparacao = df_verdadeiro[["ID", *CATEGORIAS]].merge(
        df_previsto[["ID", *CATEGORIAS]], on="ID", suffixes=("_real", "_previsto")
    )

    margem = (100 - NIVEL_CONFIANCA) / 2
    linhas = []
    for posicao, categoria in enumerate(CATEGORIAS):
        if categoria not in pontuais.index:
            continue
        validos = comparacao.dropna(subset=[f"{categoria}_previsto"])
        y_real = validos[f"{categoria}_real"].to_numpy(dtype=np.int64)
        y_previsto = validos[f"{categoria}_previsto"].to_numpy(dtype=np.int64)
        info = pontuais.loc[categoria]
        sem_positivos = info["n_positivos_gabarito"] == 0

        if not sem_positivos:
            rng = np.random.default_rng([semente, posicao])
            distribuicoes = bootstrap_metricas(y_real, y_previsto, rng, n_bootstrap)

        for metrica in METRICAS_COM_IC:
            if sem_positivos:
                estimativa = inferior = superior = np.nan
                observacao = "N/A: nenhum positivo no gabarito"
            else:
                valores = distribuicoes[metrica]
                estimativa = float(info[metrica])
                inferior, superior = (float(v) for v in np.percentile(valores, [margem, 100 - margem]))
                observacao = ""
            linhas.append(
                {
                    "modelo": nome_modelo,
                    "categoria": categoria,
                    "metrica": metrica,
                    "estimativa": estimativa,
                    "ic95_inferior": inferior,
                    "ic95_superior": superior,
                    "amplitude": superior - inferior,
                    "n_amostras": int(info["n_amostras"]),
                    "n_positivos_gabarito": int(info["n_positivos_gabarito"]),
                    "n_positivos_previstos": int(info["n_positivos_previstos"]),
                    "n_bootstrap": n_bootstrap,
                    "semente": semente,
                    "observacao": observacao,
                }
            )

    return pd.DataFrame(linhas)