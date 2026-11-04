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


# --------------------------------------------------------------------------
# Intervalo de confiança da DIFERENÇA entre duas classificações (bootstrap pareado)
# --------------------------------------------------------------------------
#
# Os intervalos acima são de cada modelo separado. Para dizer se A é melhor que B,
# o que interessa é o intervalo da diferença A − B. No bootstrap pareado, cada
# reamostragem sorteia UM conjunto de feedbacks e calcula a métrica de A e de B
# nesses mesmos feedbacks; a diferença é tirada dentro de cada reamostragem. Como
# o mesmo sorteio vale para todas as categorias, também dá para calcular a
# diferença na média GP1–GP6 em cada reamostragem.

ROTULO_MEDIA = "Média"


def _contagens(real: np.ndarray, previsto: np.ndarray):
    """VP, FP, FN e VN somando ao longo do último eixo (arrays de 0/1)."""
    vp = (real & previsto).sum(axis=-1)
    fp = ((1 - real) & previsto).sum(axis=-1)
    fn = (real & (1 - previsto)).sum(axis=-1)
    vn = real.shape[-1] - vp - fp - fn
    return vp, fp, fn, vn


def diferenca_pareada(
    df_verdadeiro: pd.DataFrame,
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    categorias: list[str],
    n_bootstrap: int = N_BOOTSTRAP,
    semente: int = SEMENTE,
) -> pd.DataFrame:
    """IC95% pareado de A − B para F1, MCC e Kappa, por categoria e na média das categorias.

    Usa só os IDs presentes no gabarito, em A e em B, e sem valor faltando em
    nenhuma das categorias pedidas (assim A e B são comparados exatamente nos
    mesmos feedbacks). As convenções de casos indefinidos são as de
    `calcular_metricas` (0).
    """
    base = (
        df_verdadeiro[["ID", *categorias]]
        .merge(df_a[["ID", *categorias]], on="ID", suffixes=("", "_a"))
        .merge(df_b[["ID", *categorias]].rename(columns={c: f"{c}_b" for c in categorias}), on="ID")
        .dropna()
    )
    n = len(base)
    real = base[categorias].to_numpy(dtype=np.int8).T  # (categorias, n); int8 economiza memória
    prev_a = base[[f"{c}_a" for c in categorias]].to_numpy(dtype=np.int8).T
    prev_b = base[[f"{c}_b" for c in categorias]].to_numpy(dtype=np.int8).T

    rng = np.random.default_rng(semente)
    indices = rng.integers(0, n, size=(n_bootstrap, n))  # o mesmo sorteio para A, B e todas as categorias

    pontual_a = metricas_das_contagens(*_contagens(real, prev_a))  # cada métrica: (categorias,)
    pontual_b = metricas_das_contagens(*_contagens(real, prev_b))
    boot_a = metricas_das_contagens(*_contagens(real[:, indices], prev_a[:, indices]))  # (categorias, n_bootstrap)
    boot_b = metricas_das_contagens(*_contagens(real[:, indices], prev_b[:, indices]))

    margem = (100 - NIVEL_CONFIANCA) / 2
    linhas = []
    for metrica in METRICAS_COM_IC:
        estimativas = list(pontual_a[metrica] - pontual_b[metrica]) + [
            float(np.mean(pontual_a[metrica]) - np.mean(pontual_b[metrica]))
        ]
        distribuicoes = list(boot_a[metrica] - boot_b[metrica]) + [
            boot_a[metrica].mean(axis=0) - boot_b[metrica].mean(axis=0)
        ]
        nomes = [*categorias, ROTULO_MEDIA]
        valores_a = list(pontual_a[metrica]) + [float(np.mean(pontual_a[metrica]))]
        valores_b = list(pontual_b[metrica]) + [float(np.mean(pontual_b[metrica]))]
        for categoria, estimativa, distribuicao, valor_a, valor_b in zip(nomes, estimativas, distribuicoes, valores_a, valores_b):
            inferior, superior = (float(v) for v in np.percentile(distribuicao, [margem, 100 - margem]))
            linhas.append(
                {
                    "categoria": categoria,
                    "metrica": metrica,
                    "valor_a": float(valor_a),
                    "valor_b": float(valor_b),
                    "diferenca": float(estimativa),
                    "ic95_inferior": inferior,
                    "ic95_superior": superior,
                    "ic_exclui_zero": bool(inferior > 0 or superior < 0),
                    "n_amostras": n,
                    "n_bootstrap": n_bootstrap,
                    "semente": semente,
                }
            )
    return pd.DataFrame(linhas)