"""
App de comparação de métricas entre os modelos (Claude, Gemini, OpenAI) na
tarefa de classificar feedbacks educacionais nas categorias GP1-GP7.

Uso:
    streamlit run app.py
"""

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.metricas import calcular_metricas
from src.modelos import MODELOS
from src.prompts import CATEGORIAS, DESCRICOES_CATEGORIAS

RAIZ = Path(__file__).resolve().parent
CAMINHO_DATASET = RAIZ / "data" / "raw" / "dataset.csv"
PASTA_PROCESSADOS = RAIZ / "data" / "processed"

NOMES_COLUNAS = {
    "modelo": "Modelo",
    "categoria": "Categoria",
    "acuracia": "Acurácia",
    "precisao": "Precisão",
    "recall": "Recall",
    "f1": "F1",
    "mcc": "MCC",
    "kappa": "Kappa",
    "n_amostras": "Nº amostras",
}

# f1/acuracia/precisao/recall vão de 0 a 1 (viram "nota" de 0 a 10 no gráfico);
# mcc/kappa vão de -1 a 1 (0 = concordância ao acaso), então o gráfico exibe o
# valor bruto em vez de uma "nota".
METRICAS_0_A_1 = ["f1", "acuracia", "precisao", "recall"]
METRICAS_MENOS1_A_1 = ["mcc", "kappa"]

st.set_page_config(page_title="Comparação de modelos - Feedback Formativo", layout="wide")


@st.cache_data
def carregar_dataset() -> pd.DataFrame:
    return pd.read_csv(CAMINHO_DATASET)


@st.cache_data
def carregar_respostas(sufixo: str = "") -> dict[str, pd.DataFrame]:
    respostas = {}
    for id_modelo in MODELOS:
        caminho = PASTA_PROCESSADOS / f"respostas_{id_modelo}{sufixo}.csv"
        if caminho.exists():
            respostas[id_modelo] = pd.read_csv(caminho)
    return respostas


def calcular_todas_metricas(respostas: dict[str, pd.DataFrame], df_gabarito: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for id_modelo, df_previsto in respostas.items():
        metricas = calcular_metricas(df_previsto, df_gabarito)
        metricas["modelo"] = MODELOS[id_modelo]["nome"]
        linhas.append(metricas)
    return pd.concat(linhas, ignore_index=True)


def grafico_por_categoria(df_metricas: pd.DataFrame, metrica: str, modelos_disponiveis: list[str], cores_disponiveis: list[str]) -> alt.Chart:
    if metrica in METRICAS_MENOS1_A_1:
        transform, dominio, titulo_eixo = f"datum.{metrica}", [-1, 1], f"{NOMES_COLUNAS[metrica]} (-1 a 1)"
    else:
        transform, dominio, titulo_eixo = f"datum.{metrica} * 10", [0, 10], f"{metrica.capitalize()} (0 a 10)"

    return (
        alt.Chart(df_metricas)
        .transform_calculate(nota=transform)
        .mark_bar()
        .encode(
            x=alt.X("categoria:N", title="Categoria"),
            xOffset=alt.XOffset("modelo:N", sort=modelos_disponiveis),
            y=alt.Y("nota:Q", title=titulo_eixo, scale=alt.Scale(domain=dominio)),
            color=alt.Color(
                "modelo:N",
                title="Modelo",
                scale=alt.Scale(domain=modelos_disponiveis, range=cores_disponiveis),
            ),
            tooltip=[
                alt.Tooltip("categoria:N", title="Categoria"),
                alt.Tooltip("modelo:N", title="Modelo"),
                alt.Tooltip("nota:Q", title=NOMES_COLUNAS.get(metrica, metrica.capitalize()), format=".2f"),
                alt.Tooltip("n_amostras:Q", title="Amostras"),
            ],
        )
        .properties(height=420)
    )


def secao_explicacoes():
    coluna_categorias, coluna_metricas = st.columns(2)

    with coluna_categorias:
        with st.expander("O que significa cada categoria (GP1-GP7)"):
            for categoria, info in DESCRICOES_CATEGORIAS.items():
                st.markdown(f"**{categoria} — {info['nome']}**: {info['definicao']}")

    with coluna_metricas:
        with st.expander("O que significa cada métrica (Acurácia, Precisão, Recall, F1, MCC, Kappa)"):
            st.markdown(
                """
Para cada categoria (ex.: GP1), a IA responde 0 ou 1 e comparamos com o que o
professor marcou no gabarito. Isso dá 4 possibilidades: **VP** (verdadeiro
positivo: os dois marcaram 1), **VN** (verdadeiro negativo: os dois marcaram
0), **FP** (falso positivo: a IA marcou 1, mas o gabarito era 0) e **FN**
(falso negativo: a IA marcou 0, mas o gabarito era 1).

- **Acurácia** — em quantas amostras a IA bateu certinho com o gabarito
  (contando tanto os 0 quanto os 1): `(VP + VN) / total`.
- **Precisão** — das vezes que a IA disse "1" (categoria presente), quantas
  realmente eram "1" no gabarito: `VP / (VP + FP)`. Mede o quanto dá pra
  confiar quando a IA aponta a categoria.
- **Recall** (revocação) — dos casos que realmente eram "1" no gabarito,
  quantos a IA conseguiu identificar: `VP / (VP + FN)`. Mede se a IA deixa
  passar casos que deveria ter marcado.
- **F1** — combina precisão e recall em um só número (média harmônica das
  duas). Só fica alto se as duas forem boas ao mesmo tempo — por isso é a
  métrica mais usada para resumir o desempenho geral.

Acurácia, Precisão, Recall e F1 têm um problema em comum: quando uma
categoria é rara (poucos "1" no gabarito, como costuma acontecer aqui), um
modelo que chuta "0" sempre já sai bem nelas sem realmente acertar nada
relevante. MCC e Kappa tentam corrigir essa distorção:

- **MCC** (Coeficiente de Correlação de Matthews) — usa as 4 possibilidades
  (VP, VN, FP, FN) de uma vez e mede o quão correlacionadas estão as
  previsões da IA com o gabarito, de forma parecida com uma correlação
  estatística. Vai de **-1** (a IA sempre erra, discordância total) a **+1**
  (a IA acerta tudo), passando por **0** (previsões não relacionadas ao
  gabarito, como um chute aleatório). Como considera VP, VN, FP e FN juntos,
  não é enganado por categorias desbalanceadas — é considerada uma das
  métricas mais confiáveis para esse cenário.
- **Kappa de Cohen** — mede a concordância entre a IA e o professor
  descontando a concordância que aconteceria só por acaso (por exemplo, se
  os dois costumam marcar "0" na maioria das vezes, uma boa parte da
  concordância já era esperada mesmo sem nenhuma capacidade real de
  classificar). Também vai de **-1** a **+1**, onde **1** é concordância
  perfeita, **0** é o mesmo nível de concordância que dois avaliadores
  aleatórios teriam, e valores negativos indicam concordância pior que o
  acaso.

Quando o gabarito não tem nenhum exemplo "1" para uma categoria (como GP7 no
nosso dataset) e a IA também só prevê "0": Precisão, Recall e F1 não têm o
que medir e ficam em 0. Já a **Acurácia** fica enganosamente alta (perto de
1, já que os dois "acertam" o 0 quase sempre) mesmo sem a IA ter mostrado
nenhuma capacidade real de identificar a categoria — é exatamente esse tipo
de distorção que MCC e Kappa evitam: como não têm nenhuma variação real pra
medir (VP+FP+FN = 0), o cálculo original também fica indefinido (0/0), e
tratamos como 0 — mostrando corretamente que não houve concordância
significativa, ao contrário do que a Acurácia sozinha sugeriria.
                """
            )


def secao_tabelas(df_metricas: pd.DataFrame, modelos_disponiveis: list[str]):
    st.subheader("Resumo geral (média das 7 categorias)")
    resumo = (
        df_metricas.groupby("modelo")[["acuracia", "precisao", "recall", "f1", "mcc", "kappa"]]
        .mean()
        .reindex(modelos_disponiveis)
        .round(2)
    )
    st.dataframe(resumo.rename(columns=NOMES_COLUNAS), use_container_width=True)

    st.subheader("Métricas detalhadas por categoria")
    tabela = df_metricas[["modelo", "categoria", "acuracia", "precisao", "recall", "f1", "mcc", "kappa", "n_amostras"]].round(2)
    st.dataframe(tabela.rename(columns=NOMES_COLUNAS), use_container_width=True, hide_index=True)


def secao_respostas_por_modelo(df_gabarito: pd.DataFrame, respostas: dict[str, pd.DataFrame]):
    st.subheader("Respostas dos modelos (formato do dataset)")
    abas = st.tabs([MODELOS[id_modelo]["nome"] for id_modelo in respostas])
    for aba, (id_modelo, df_previsto) in zip(abas, respostas.items()):
        with aba:
            tabela_modelo = df_gabarito[["ID", "Text"]].merge(df_previsto[["ID", *CATEGORIAS]], on="ID")
            st.dataframe(tabela_modelo, use_container_width=True, hide_index=True)


def main():
    st.title("Comparação de modelos na classificação de feedbacks")
    st.caption(
        "Compara as respostas de Claude, Gemini e OpenAI com o gabarito humano do "
        "dataset, categoria por categoria (GP1-GP7)."
    )

    usar_teste = st.sidebar.checkbox("Usar resultados de teste (gerados com --n)", value=False)
    sufixo = "_teste" if usar_teste else ""
    if usar_teste:
        st.caption("🧪 Mostrando resultados de teste (arquivos `*_teste.csv`), não a rodada completa.")

    df_gabarito = carregar_dataset()
    respostas = carregar_respostas(sufixo)

    if not respostas:
        comando = "python scripts/gerar_respostas.py --n 10" if usar_teste else "python scripts/gerar_respostas.py"
        st.warning(f"Nenhum resultado encontrado em `data/processed/`. Rode primeiro:\n\n```\n{comando}\n```")
        st.stop()

    modelos_faltando = [id_modelo for id_modelo in MODELOS if id_modelo not in respostas]
    if modelos_faltando:
        nomes = ", ".join(MODELOS[id_modelo]["nome"] for id_modelo in modelos_faltando)
        st.info(f"Sem resultados para: {nomes}. Rode `scripts/gerar_respostas.py` com as chaves de API correspondentes configuradas.")

    df_metricas = calcular_todas_metricas(respostas, df_gabarito)
    modelos_disponiveis = [MODELOS[id_modelo]["nome"] for id_modelo in respostas]
    cores_disponiveis = [MODELOS[id_modelo]["cor"] for id_modelo in respostas]

    st.subheader("Métrica por categoria")
    metrica_escolhida = st.selectbox(
        "Métrica",
        ["f1", "acuracia", "precisao", "recall", "mcc", "kappa"],
        format_func=lambda m: NOMES_COLUNAS.get(m, m.capitalize()),
    )
    grafico = grafico_por_categoria(df_metricas, metrica_escolhida, modelos_disponiveis, cores_disponiveis)
    st.altair_chart(grafico, use_container_width=True)

    secao_explicacoes()
    secao_tabelas(df_metricas, modelos_disponiveis)
    secao_respostas_por_modelo(df_gabarito, respostas)


if __name__ == "__main__":
    main()
