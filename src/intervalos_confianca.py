<<<<<<< HEAD
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

from itertools import combinations

=======
>>>>>>> 130aa4bddc1abb2ed123e6fcd710bf1f9e6ff2c5
import numpy as np
import pandas as pd

from src.metricas import calcular_metricas
from src.prompts import CATEGORIAS

N_BOOTSTRAP = 1000
SEMENTE = 42
NIVEL_CONFIANCA = 95
METRICAS_COM_IC = ["f1", "mcc", "kappa"]


def _dividir(numerador: np.ndarray, denominador: np.ndarray) -> np.ndarray:
    resultado = np.zeros_like(numerador, dtype=float)
    np.divide(numerador, denominador, out=resultado, where=denominador != 0)
    return resultado


def metricas_das_contagens(vp, fp, fn, vn) -> dict[str, np.ndarray]:
    vp, fp, fn, vn = (np.asarray(x, dtype=float) for x in (vp, fp, fn, vn))
    total = vp + fp + fn + vn

    f1 = _dividir(2 * vp, 2 * vp + fp + fn)

    denominador_mcc = np.sqrt((vp + fp) * (vp + fn) * (vn + fp) * (vn + fn))
    mcc = _dividir(vp * vn - fp * fn, denominador_mcc)

    concordancia_observada = _dividir(vp + vn, total)
    concordancia_acaso = _dividir((vp + fp) * (vp + fn) + (vn + fn) * (vn + fp), total**2)
    kappa = _dividir(concordancia_observada - concordancia_acaso, 1 - concordancia_acaso)

    return {"f1": f1, "mcc": mcc, "kappa": kappa}


<<<<<<< HEAD
def metricas_reamostradas(y_real: np.ndarray, y_previsto: np.ndarray, indices: np.ndarray) -> dict[str, np.ndarray]:
    """F1, MCC e Kappa em cada reamostragem (uma linha de `indices` por reamostragem)."""
    n = indices.shape[1]
=======
def bootstrap_metricas(
    y_real: np.ndarray, y_previsto: np.ndarray, rng: np.random.Generator, n_bootstrap: int = N_BOOTSTRAP
) -> dict[str, np.ndarray]:
    n = len(y_real)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
>>>>>>> 130aa4bddc1abb2ed123e6fcd710bf1f9e6ff2c5
    real = y_real[indices]
    previsto = y_previsto[indices]

    vp = (real & previsto).sum(axis=1)
    fp = ((1 - real) & previsto).sum(axis=1)
    fn = (real & (1 - previsto)).sum(axis=1)
    vn = n - vp - fp - fn
    return metricas_das_contagens(vp, fp, fn, vn)


def bootstrap_metricas(
    y_real: np.ndarray, y_previsto: np.ndarray, rng: np.random.Generator, n_bootstrap: int = N_BOOTSTRAP
) -> dict[str, np.ndarray]:
    """Distribuição bootstrap (n_bootstrap valores) de F1, MCC e Kappa."""
    n = len(y_real)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    return metricas_reamostradas(y_real, y_previsto, indices)


def calcular_ic95(
    df_previsto: pd.DataFrame,
    df_verdadeiro: pd.DataFrame,
    nome_modelo: str,
    n_bootstrap: int = N_BOOTSTRAP,
    semente: int = SEMENTE,
) -> pd.DataFrame:
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


<<<<<<< HEAD
def calcular_ic95_diferencas(
    previsoes: dict[str, pd.DataFrame],
    df_verdadeiro: pd.DataFrame,
    n_bootstrap: int = N_BOOTSTRAP,
    semente: int = SEMENTE,
) -> pd.DataFrame:
    """IC95% da DIFERENÇA de F1, MCC e Kappa entre cada par de modelos.

    previsoes: {nome do modelo: DataFrame de respostas}, na ordem de exibição.

    Para dizer "o modelo A é melhor que o B no GP1" não basta olhar se os
    intervalos de cada um se sobrepõem: o certo é o intervalo da diferença
    A − B. Como todos os modelos responderam os MESMOS feedbacks, cada
    reamostragem sorteia os mesmos feedbacks para todos (bootstrap pareado) e a
    diferença é calculada dentro de cada reamostragem. Se o intervalo não
    contém 0, a diferença é estatisticamente significativa (nível de 5%).

    Só entram os feedbacks que todos os modelos responderam naquela categoria,
    para que a comparação seja sempre sobre o mesmo conjunto. Sem nenhuma falha
    de parsing, as reamostragens são exatamente as de `calcular_ic95`.
    """
    nomes = list(previsoes)
    comparacao = df_verdadeiro[["ID", *CATEGORIAS]]
    for nome, df_previsto in previsoes.items():
        comparacao = comparacao.merge(
            df_previsto[["ID", *CATEGORIAS]].rename(columns={c: f"{c}|{nome}" for c in CATEGORIAS}), on="ID"
        )

    margem = (100 - NIVEL_CONFIANCA) / 2
    linhas = []
    for posicao, categoria in enumerate(CATEGORIAS):
        colunas_modelos = [f"{categoria}|{nome}" for nome in nomes]
        validos = comparacao.dropna(subset=colunas_modelos)
        if validos.empty:
            continue
        y_real = validos[categoria].to_numpy(dtype=np.int64)
        n_positivos = int(y_real.sum())
        sem_positivos = n_positivos == 0

        if not sem_positivos:
            n = len(y_real)
            todos = np.arange(n)[np.newaxis, :]  # a amostra original, para a estimativa pontual
            indices = np.random.default_rng([semente, posicao]).integers(0, n, size=(n_bootstrap, n))
            pontuais, distribuicoes = {}, {}
            for nome, coluna in zip(nomes, colunas_modelos):
                y_previsto = validos[coluna].to_numpy(dtype=np.int64)
                pontuais[nome] = metricas_reamostradas(y_real, y_previsto, todos)
                distribuicoes[nome] = metricas_reamostradas(y_real, y_previsto, indices)

        for modelo_a, modelo_b in combinations(nomes, 2):
            for metrica in METRICAS_COM_IC:
                if sem_positivos:
                    diferenca = inferior = superior = np.nan
                    conclusao = "N/A"
                    observacao = "N/A: nenhum positivo no gabarito"
                else:
                    diferenca = float(pontuais[modelo_a][metrica][0] - pontuais[modelo_b][metrica][0])
                    valores = distribuicoes[modelo_a][metrica] - distribuicoes[modelo_b][metrica]
                    inferior, superior = (float(v) for v in np.percentile(valores, [margem, 100 - margem]))
                    if inferior > 0:
                        conclusao = f"{modelo_a} melhor"
                    elif superior < 0:
                        conclusao = f"{modelo_b} melhor"
                    else:
                        conclusao = "sem diferença significativa"
                    observacao = ""
                linhas.append(
                    {
                        "categoria": categoria,
                        "metrica": metrica,
                        "modelo_a": modelo_a,
                        "modelo_b": modelo_b,
                        "diferenca": diferenca,
                        "ic95_inferior": inferior,
                        "ic95_superior": superior,
                        "conclusao": conclusao,
                        "n_amostras": len(validos),
                        "n_positivos_gabarito": n_positivos,
                        "n_bootstrap": n_bootstrap,
                        "semente": semente,
                        "observacao": observacao,
                    }
                )

=======
ROTULO_MEDIA = "Média"


def _contagens(real: np.ndarray, previsto: np.ndarray):
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
    base = (
        df_verdadeiro[["ID", *categorias]]
        .merge(df_a[["ID", *categorias]], on="ID", suffixes=("", "_a"))
        .merge(df_b[["ID", *categorias]].rename(columns={c: f"{c}_b" for c in categorias}), on="ID")
        .dropna()
    )
    n = len(base)
    real = base[categorias].to_numpy(dtype=np.int8).T
    prev_a = base[[f"{c}_a" for c in categorias]].to_numpy(dtype=np.int8).T
    prev_b = base[[f"{c}_b" for c in categorias]].to_numpy(dtype=np.int8).T

    rng = np.random.default_rng(semente)
    indices = rng.integers(0, n, size=(n_bootstrap, n))

    pontual_a = metricas_das_contagens(*_contagens(real, prev_a))
    pontual_b = metricas_das_contagens(*_contagens(real, prev_b))
    boot_a = metricas_das_contagens(*_contagens(real[:, indices], prev_a[:, indices]))
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
>>>>>>> 130aa4bddc1abb2ed123e6fcd710bf1f9e6ff2c5
    return pd.DataFrame(linhas)
