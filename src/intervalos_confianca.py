from itertools import combinations

import numpy as np
import pandas as pd

from src.metricas import calcular_metricas
from src.prompts import CATEGORIAS

N_BOOTSTRAP = 1000
SEMENTE = 42
NIVEL_CONFIANCA = 95
METRICAS_COM_IC = ["f1", "mcc", "kappa"]
ROTULO_MEDIA = "Média"


def _dividir(numerador: np.ndarray, denominador: np.ndarray) -> np.ndarray:
    resultado = np.zeros_like(numerador, dtype=float)
    np.divide(numerador, denominador, out=resultado, where=denominador != 0)
    return resultado


def _percentis(valores: np.ndarray) -> tuple[float, float]:
    margem = (100 - NIVEL_CONFIANCA) / 2
    inferior, superior = np.percentile(valores, [margem, 100 - margem])
    return float(inferior), float(superior)


def _contagens(real: np.ndarray, previsto: np.ndarray):
    vp = (real & previsto).sum(axis=-1)
    fp = ((1 - real) & previsto).sum(axis=-1)
    fn = (real & (1 - previsto)).sum(axis=-1)
    vn = real.shape[-1] - vp - fp - fn
    return vp, fp, fn, vn


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


def bootstrap_metricas(
    y_real: np.ndarray, y_previsto: np.ndarray, rng: np.random.Generator, n_bootstrap: int = N_BOOTSTRAP
) -> dict[str, np.ndarray]:
    indices = rng.integers(0, len(y_real), size=(n_bootstrap, len(y_real)))
    return metricas_das_contagens(*_contagens(y_real[indices], y_previsto[indices]))


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
                inferior, superior = _percentis(valores)
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


def _conclusao(inferior: float, superior: float, nome_a: str, nome_b: str) -> str:
    if inferior > 0:
        return f"{nome_a} melhor"
    if superior < 0:
        return f"{nome_b} melhor"
    return "sem diferença significativa"


def diferenca_pareada(
    df_verdadeiro: pd.DataFrame,
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    categorias: list[str],
    nome_a: str = "A",
    nome_b: str = "B",
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
    positivos = [*(int(p) for p in real.sum(axis=1)), int(real.sum())]

    indices = np.random.default_rng(semente).integers(0, n, size=(n_bootstrap, n))
    pontual_a = metricas_das_contagens(*_contagens(real, prev_a))
    pontual_b = metricas_das_contagens(*_contagens(real, prev_b))
    boot_a = metricas_das_contagens(*_contagens(real[:, indices], prev_a[:, indices]))
    boot_b = metricas_das_contagens(*_contagens(real[:, indices], prev_b[:, indices]))

    linhas = []
    for metrica in METRICAS_COM_IC:
        valores_a = [*pontual_a[metrica], pontual_a[metrica].mean()]
        valores_b = [*pontual_b[metrica], pontual_b[metrica].mean()]
        distribuicoes = [*(boot_a[metrica] - boot_b[metrica]), boot_a[metrica].mean(axis=0) - boot_b[metrica].mean(axis=0)]
        for categoria, valor_a, valor_b, distribuicao, n_positivos in zip(
            [*categorias, ROTULO_MEDIA], valores_a, valores_b, distribuicoes, positivos
        ):
            inferior, superior = _percentis(distribuicao)
            linhas.append(
                {
                    "categoria": categoria,
                    "metrica": metrica,
                    "valor_a": float(valor_a),
                    "valor_b": float(valor_b),
                    "diferenca": float(valor_a - valor_b),
                    "ic95_inferior": inferior,
                    "ic95_superior": superior,
                    "ic_exclui_zero": bool(inferior > 0 or superior < 0),
                    "conclusao": _conclusao(inferior, superior, nome_a, nome_b),
                    "n_amostras": n,
                    "n_positivos_gabarito": n_positivos,
                    "n_bootstrap": n_bootstrap,
                    "semente": semente,
                }
            )
    return pd.DataFrame(linhas)


def calcular_ic95_diferencas(previsoes: dict[str, pd.DataFrame], df_verdadeiro: pd.DataFrame) -> pd.DataFrame:
    categorias = [c for c in CATEGORIAS if df_verdadeiro[c].sum() > 0]
    resultados = []
    for nome_a, nome_b in combinations(previsoes, 2):
        df = diferenca_pareada(df_verdadeiro, previsoes[nome_a], previsoes[nome_b], categorias, nome_a, nome_b)
        df.insert(0, "modelo_b", nome_b)
        df.insert(0, "modelo_a", nome_a)
        resultados.append(df)
    return pd.concat(resultados, ignore_index=True)
