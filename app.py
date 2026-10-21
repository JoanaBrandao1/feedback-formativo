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

from src.metricas import (
    COLUNAS_METRICAS,
    METRICAS_SEM_POSITIVOS_INDEFINIDAS,
    calcular_metricas,
    marcar_nao_definidas,
    media_por_modelo,
    resumir_falhas,
)
from src.modelos import MODELOS
from src.prompts import CATEGORIAS, DESCRICOES_CATEGORIAS

RAIZ = Path(__file__).resolve().parent
CAMINHO_DATASET = RAIZ / "data" / "raw" / "dataset.csv"
PASTA_PROCESSADOS = RAIZ / "data" / "processed"
CAMINHO_IC95 = PASTA_PROCESSADOS / "metricas_ic95.csv"  # gerado por scripts/calcular_ic95.py
CAMINHO_CONSISTENCIA = PASTA_PROCESSADOS / "consistencia_claude_resumo.csv"  # gerado por scripts/teste_consistencia.py

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
    "n_positivos_gabarito": "Nº de positivos no gabarito",
    "n_positivos_previstos": "Nº de positivos previstos pelo modelo",
}

# f1/acuracia/precisao/recall vão de 0 a 1 (viram "nota" de 0 a 10 no gráfico);
# mcc/kappa vão de -1 a 1 (0 = concordância ao acaso), então o gráfico exibe o
# valor bruto em vez de uma "nota".
METRICAS_0_A_1 = ["f1", "acuracia", "precisao", "recall"]
METRICAS_MENOS1_A_1 = ["mcc", "kappa"]
METRICAS_COM_IC = ["f1", "mcc", "kappa"]

# Abaixo disso, o gabarito tem tão poucos positivos que as métricas daquela
# categoria são muito instáveis (intervalos de confiança largos).
LIMITE_POUCOS_POSITIVOS = 50

# No teste de consistência (amostra pequena), com menos positivos que isso em
# ambas as execuções a concordância só reflete o "0" e não diz nada sobre a categoria.
LIMITE_POUCOS_POSITIVOS_CONSISTENCIA = 5

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


def rotulo_categorias(categorias: list[str]) -> str:
    """["GP1", ..., "GP6"] -> "GP1–GP6"; se não forem consecutivas, lista todas."""
    if not categorias:
        return ""
    numeros = [CATEGORIAS.index(c) for c in categorias]
    if numeros == list(range(numeros[0], numeros[-1] + 1)) and len(numeros) > 1:
        return f"{categorias[0]}–{categorias[-1]}"
    return ", ".join(categorias)


def grafico_por_categoria(df_metricas: pd.DataFrame, metrica: str, modelos_disponiveis: list[str], cores_disponiveis: list[str]) -> alt.Chart:
    if metrica in METRICAS_MENOS1_A_1:
        transform, dominio, titulo_eixo = f"datum.{metrica}", [-1, 1], f"{NOMES_COLUNAS[metrica]} (-1 a 1)"
    else:
        transform, dominio, titulo_eixo = f"datum.{metrica} * 10", [0, 10], f"{metrica.capitalize()} (0 a 10)"

    # Categoria sem positivos no gabarito: a métrica não existe (N/A), então a
    # barra some e o rótulo do eixo avisa, em vez de desenhar uma barra "0".
    df_grafico = df_metricas
    if metrica in METRICAS_SEM_POSITIVOS_INDEFINIDAS:
        df_grafico = marcar_nao_definidas(df_metricas)
    df_grafico = df_grafico.assign(
        rotulo=[
            f"{categoria} (N/A)" if pd.isna(valor) else categoria
            for categoria, valor in zip(df_grafico["categoria"], df_grafico[metrica])
        ]
    )
    ordem_rotulos = (
        df_grafico.drop_duplicates("categoria")
        .set_index("categoria")
        .loc[[c for c in CATEGORIAS if c in set(df_grafico["categoria"])], "rotulo"]
        .tolist()
    )

    return (
        alt.Chart(df_grafico)
        .transform_calculate(nota=transform)
        .transform_filter(f"isValid(datum.{metrica})")
        .mark_bar()
        .encode(
            x=alt.X("rotulo:N", title="Categoria", sort=ordem_rotulos),
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
                alt.Tooltip("n_positivos_gabarito:Q", title="Positivos no gabarito"),
            ],
        )
        .properties(height=420)
    )


def grafico_ic95(df_ic: pd.DataFrame, metrica: str, modelos_disponiveis: list[str], cores_disponiveis: list[str]) -> alt.LayerChart:
    """Barras com a estimativa + linha vertical com o IC de 95% (bootstrap)."""
    dados = df_ic[(df_ic["metrica"] == metrica) & df_ic["estimativa"].notna()]
    ordem = [c for c in CATEGORIAS if c in set(dados["categoria"])]
    dominio = [-1, 1] if metrica in METRICAS_MENOS1_A_1 else [0, 1]

    x = alt.X("categoria:N", title="Categoria", sort=ordem)
    x_offset = alt.XOffset("modelo:N", sort=modelos_disponiveis)
    escala_y = alt.Scale(domain=dominio)
    cor = alt.Color(
        "modelo:N",
        title="Modelo",
        scale=alt.Scale(domain=modelos_disponiveis, range=cores_disponiveis),
    )
    tooltip = [
        alt.Tooltip("categoria:N", title="Categoria"),
        alt.Tooltip("modelo:N", title="Modelo"),
        alt.Tooltip("estimativa:Q", title=NOMES_COLUNAS[metrica], format=".2f"),
        alt.Tooltip("ic95_inferior:Q", title="IC95% inferior", format=".2f"),
        alt.Tooltip("ic95_superior:Q", title="IC95% superior", format=".2f"),
        alt.Tooltip("n_positivos_gabarito:Q", title="Positivos no gabarito"),
    ]

    base = alt.Chart(dados)
    barras = base.mark_bar(opacity=0.45).encode(
        x=x, xOffset=x_offset, color=cor, tooltip=tooltip,
        y=alt.Y("estimativa:Q", title=f"{NOMES_COLUNAS[metrica]} (barra) e IC95% (linha)", scale=escala_y),
    )
    intervalos = base.mark_rule(strokeWidth=2).encode(
        x=x, xOffset=x_offset, color=cor, tooltip=tooltip,
        y=alt.Y("ic95_inferior:Q", scale=escala_y),
        y2="ic95_superior:Q",
    )
    return (barras + intervalos).properties(height=420)


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
nosso dataset), **Recall, F1, MCC e Kappa não têm o que medir** (não existe
nenhum caso positivo para a IA encontrar) e aparecem como **N/A** (não
definido), e não como 0. A **Precisão** também aparece como N/A: se a IA
marcar a categoria algumas vezes, todas essas marcações são falsos positivos
(precisão 0), mas isso não diz nada sobre a capacidade de identificar a
categoria, já que ela nunca está presente. A **Acurácia** continua aparecendo,
mas fica enganosamente alta (perto de 1) só porque a IA marca "0" quase
sempre — por isso ela vem com uma nota. Pelo mesmo motivo, o resumo geral usa
só as categorias que têm positivos no gabarito; a média que inclui essas
categorias (contando as métricas como 0) fica numa tabela à parte, só como
referência.
                """
            )


def categorias_sem_positivos(df_metricas: pd.DataFrame) -> list[str]:
    sem_positivos = set(df_metricas.loc[df_metricas["n_positivos_gabarito"] == 0, "categoria"])
    return [c for c in CATEGORIAS if c in sem_positivos]


def nota_sem_positivos(df_metricas: pd.DataFrame, sem_positivos: list[str]) -> str:
    """Texto que explica por que algumas categorias aparecem como N/A."""
    partes = []
    for categoria in sem_positivos:
        previstos = df_metricas[df_metricas["categoria"] == categoria].set_index("modelo")["n_positivos_previstos"]
        marcacoes = ", ".join(f"{modelo}: {n}" for modelo, n in previstos.items())
        partes.append(
            f"**{categoria}** não tem nenhum exemplo positivo (1) no gabarito, então recall, F1, MCC e Kappa "
            f"não estão definidos e a precisão não é interpretável — por isso aparecem como **N/A**, e não 0. "
            f"A **acurácia** de {categoria} continua exibida, mas fica alta só porque os modelos marcam 0 quase "
            f"sempre; ela não mostra capacidade de identificar a categoria. "
            f"Vezes em que cada modelo marcou {categoria}=1 (todas falsos positivos): {marcacoes}."
        )
    return "\n\n".join(partes)


def secao_tabelas(df_metricas: pd.DataFrame, modelos_disponiveis: list[str]):
    sem_positivos = categorias_sem_positivos(df_metricas)
    com_positivos = [c for c in CATEGORIAS if c in set(df_metricas["categoria"]) and c not in sem_positivos]

    st.subheader(f"Resumo geral (média das categorias com positivos no gabarito: {rotulo_categorias(com_positivos)})")
    resumo = media_por_modelo(df_metricas, modelos_disponiveis).round(2)
    st.dataframe(resumo.rename(columns=NOMES_COLUNAS), width="stretch")

    if sem_positivos:
        with st.expander(f"Média com {', '.join(sem_positivos)} (referência)"):
            st.caption(
                f"Média antiga, de todas as categorias, contando {', '.join(sem_positivos)} com precisão, recall, "
                "F1, MCC e Kappa = 0. Não é comparável com a tabela acima e é mostrada só como referência."
            )
            resumo_com_todas = media_por_modelo(df_metricas, modelos_disponiveis, somente_com_positivos=False).round(2)
            st.dataframe(resumo_com_todas.rename(columns=NOMES_COLUNAS), width="stretch")

    st.subheader("Métricas detalhadas por categoria")
    colunas = ["modelo", "categoria", *COLUNAS_METRICAS, "n_amostras", "n_positivos_gabarito", "n_positivos_previstos"]
    tabela = marcar_nao_definidas(df_metricas)[colunas].rename(columns=NOMES_COLUNAS)
    colunas_metricas = [NOMES_COLUNAS[c] for c in COLUNAS_METRICAS]
    tabela_formatada = tabela.style.format("{:.2f}", subset=colunas_metricas, na_rep="N/A")
    st.dataframe(tabela_formatada, width="stretch", hide_index=True)
    if sem_positivos:
        st.caption(nota_sem_positivos(df_metricas, sem_positivos))


def secao_intervalos_confianca(modelos_disponiveis: list[str], cores_disponiveis: list[str], usar_teste: bool):
    st.subheader("Intervalos de confiança de 95% (bootstrap)")
    if usar_teste:
        st.info("Os intervalos de confiança só existem para a rodada completa. Desmarque \"Usar resultados de teste\" para vê-los.")
        return
    if not CAMINHO_IC95.exists():
        st.info("Ainda não há intervalos de confiança. Gere-os (sem chamar nenhuma API) com:\n\n```\npython scripts/calcular_ic95.py\n```")
        return

    df_ic = pd.read_csv(CAMINHO_IC95)
    df_ic = df_ic[df_ic["modelo"].isin(modelos_disponiveis)]
    n_bootstrap, semente = int(df_ic["n_bootstrap"].iloc[0]), int(df_ic["semente"].iloc[0])
    st.caption(
        f"Para cada modelo e categoria, o conjunto de {df_ic['n_amostras'].max()} feedbacks é reamostrado "
        f"{n_bootstrap} vezes (com reposição, semente fixa {semente}) e a métrica é recalculada a cada vez; "
        "o intervalo vai do percentil 2,5% ao 97,5%. Quanto mais largo o intervalo, menos confiável é a "
        "estimativa. Se os intervalos de dois modelos se sobrepõem bastante, a diferença entre eles pode ser "
        "só ruído da amostra."
    )

    # categorias com poucos positivos → intervalos largos
    por_categoria = (
        df_ic[df_ic["metrica"] == "f1"]
        .groupby("categoria")
        .agg(amplitude=("amplitude", "mean"), positivos=("n_positivos_gabarito", "first"))
    )
    poucos = por_categoria[(por_categoria["positivos"] > 0) & (por_categoria["positivos"] < LIMITE_POUCOS_POSITIVOS)]
    demais = por_categoria[(por_categoria["positivos"] >= LIMITE_POUCOS_POSITIVOS)]
    if not poucos.empty and not demais.empty:
        lista = " e ".join(f"{c} ({int(r.positivos)} positivos, amplitude média do IC do F1 = {r.amplitude:.2f})" for c, r in poucos.iterrows())
        st.warning(
            f"⚠️ {lista} têm poucos positivos no gabarito, por isso seus intervalos são bem mais largos "
            f"(nas demais categorias com positivos, a amplitude média fica entre {demais['amplitude'].min():.2f} e "
            f"{demais['amplitude'].max():.2f}). As diferenças entre os modelos nessas categorias devem ser lidas com cautela."
        )

    metrica_ic = st.selectbox(
        "Métrica",
        METRICAS_COM_IC,
        format_func=lambda m: NOMES_COLUNAS[m],
        key="metrica_ic",
    )
    st.altair_chart(grafico_ic95(df_ic, metrica_ic, modelos_disponiveis, cores_disponiveis), width="stretch")
    sem_positivos = sorted(set(df_ic.loc[df_ic["estimativa"].isna(), "categoria"]))
    if sem_positivos:
        st.caption(f"{', '.join(sem_positivos)} não aparece no gráfico: sem nenhum positivo no gabarito, a métrica é N/A.")

    def formatar(linha: pd.Series) -> str:
        if pd.isna(linha["estimativa"]):
            return "N/A"
        return f"{linha['estimativa']:.2f}  [{linha['ic95_inferior']:.2f} ; {linha['ic95_superior']:.2f}]"

    df_ic = df_ic.assign(texto=df_ic.apply(formatar, axis=1))
    tabela = df_ic.pivot_table(
        index=["modelo", "categoria", "n_positivos_gabarito"], columns="metrica", values="texto", aggfunc="first"
    ).reset_index()
    tabela["modelo"] = pd.Categorical(tabela["modelo"], categories=modelos_disponiveis, ordered=True)
    tabela["categoria"] = pd.Categorical(tabela["categoria"], categories=CATEGORIAS, ordered=True)
    tabela = tabela.sort_values(["modelo", "categoria"])
    tabela["Aviso"] = tabela["n_positivos_gabarito"].map(
        lambda n: "⚠️ poucos positivos" if 0 < n < LIMITE_POUCOS_POSITIVOS else ""
    )
    tabela = tabela[["modelo", "categoria", "n_positivos_gabarito", *METRICAS_COM_IC, "Aviso"]].rename(
        columns={**NOMES_COLUNAS, **{m: f"{NOMES_COLUNAS[m]} [IC 95%]" for m in METRICAS_COM_IC}}
    )
    st.dataframe(tabela, width="stretch", hide_index=True)


def secao_qualidade_respostas(respostas: dict[str, pd.DataFrame]):
    st.subheader("Falhas de parsing e respostas incompletas")
    linhas = []
    for id_modelo, df_previsto in respostas.items():
        falhas = resumir_falhas(df_previsto)
        linhas.append(
            {
                "Modelo": MODELOS[id_modelo]["nome"],
                "Feedbacks": falhas["total"],
                "Respostas vazias (API falhou)": falhas["resposta_vazia"],
                "Feedbacks com algum GP sem valor": falhas["com_gp_nulo"],
            }
        )
    tabela = pd.DataFrame(linhas)
    st.dataframe(tabela, width="stretch", hide_index=True)
    if tabela["Feedbacks com algum GP sem valor"].sum() == 0:
        st.caption("Nenhum modelo teve falha: todas as respostas foram lidas por completo, então todas as métricas usam os mesmos feedbacks.")
    else:
        st.warning(
            "Há respostas com falha. Elas ficam de fora do cálculo da(s) categoria(s) afetada(s) só para aquele "
            "modelo, então os modelos deixam de ser comparados exatamente sobre os mesmos feedbacks."
        )


def secao_consistencia():
    st.subheader("Consistência do Claude entre duas execuções")
    if not CAMINHO_CONSISTENCIA.exists():
        st.info(
            "Ainda não há teste de consistência. Ele refaz parte das perguntas ao Claude e **faz chamadas pagas à "
            "API** (veja o README, seção 4); por isso não roda sozinho."
        )
        return

    resumo = pd.read_csv(CAMINHO_CONSISTENCIA)
    todas = resumo[resumo["categoria"].str.startswith("TODAS")].iloc[0]
    por_categoria = resumo[~resumo["categoria"].str.startswith("TODAS")].copy()
    n_feedbacks = int(todas["n_comparados"])

    st.caption(
        "O Claude roda sem `temperature=0` (a API não aceita esse parâmetro para ele), então pode responder diferente "
        f"da mesma pergunta. Por isso {n_feedbacks} feedbacks sorteados (semente fixa) foram perguntados de novo, "
        "com o mesmo prompt, e a nova resposta foi comparada com a da rodada original."
    )

    coluna_a, coluna_b = st.columns(2)
    coluna_a.metric(
        "Feedbacks com os 7 GP idênticos nas 2 execuções",
        f"{todas['pct_concordancia']:.0f}%",
        help=f"{round(todas['pct_concordancia'] * n_feedbacks / 100)} de {n_feedbacks} feedbacks.",
    )
    coluna_b.metric(
        "Feedbacks em que algum GP mudou",
        f"{round((100 - todas['pct_concordancia']) * n_feedbacks / 100)} de {n_feedbacks}",
    )

    por_categoria["poucos_positivos"] = (
        por_categoria[["positivos_original", "positivos_nova"]].max(axis=1) < LIMITE_POUCOS_POSITIVOS_CONSISTENCIA
    )
    por_categoria["Aviso"] = por_categoria["poucos_positivos"].map(
        lambda poucos: "⚠️ quase sem positivos: só reflete o 0" if poucos else ""
    )

    cor_claude = MODELOS["claude"]["cor"]
    grafico = (
        alt.Chart(por_categoria)
        .mark_bar()
        .encode(
            x=alt.X("categoria:N", title="Categoria", sort=CATEGORIAS),
            y=alt.Y("pct_concordancia:Q", title="Concordância entre as 2 execuções (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.condition(
                alt.datum.poucos_positivos,
                alt.value("#b0b0b0"),
                alt.value(cor_claude),
            ),
            tooltip=[
                alt.Tooltip("categoria:N", title="Categoria"),
                alt.Tooltip("pct_concordancia:Q", title="Concordância (%)", format=".1f"),
                alt.Tooltip("kappa:Q", title="Kappa", format=".2f"),
                alt.Tooltip("positivos_original:Q", title="Positivos (original)"),
                alt.Tooltip("positivos_nova:Q", title="Positivos (nova)"),
            ],
        )
        .properties(height=300)
    )
    st.altair_chart(grafico, width="stretch")
    st.caption("Barras cinzas: categorias com quase nenhum positivo na amostra, onde a concordância só mostra que as duas execuções marcaram 0.")

    tabela = por_categoria.rename(
        columns={
            "categoria": "Categoria",
            "n_comparados": "Feedbacks comparados",
            "pct_concordancia": "Concordância (%)",
            "kappa": "Kappa",
            "positivos_original": "Positivos (rodada original)",
            "positivos_nova": "Positivos (nova rodada)",
        }
    )[["Categoria", "Feedbacks comparados", "Concordância (%)", "Kappa", "Positivos (rodada original)", "Positivos (nova rodada)", "Aviso"]]
    st.dataframe(
        tabela.style.format({"Concordância (%)": "{:.1f}", "Kappa": "{:.2f}"}, na_rep="N/A"),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Kappa N/A: indefinido, porque as duas execuções marcaram sempre o mesmo valor (só 0). É apenas uma "
        "repetição, em uma amostra pequena e só do Claude; Gemini e OpenAI usam `temperature=0` e não foram "
        "testados. Esta variação é uma incerteza a mais, que não está nos intervalos de confiança acima."
    )


def secao_respostas_por_modelo(df_gabarito: pd.DataFrame, respostas: dict[str, pd.DataFrame]):
    st.subheader("Respostas dos modelos (formato do dataset)")
    abas = st.tabs([MODELOS[id_modelo]["nome"] for id_modelo in respostas])
    for aba, (id_modelo, df_previsto) in zip(abas, respostas.items()):
        with aba:
            tabela_modelo = df_gabarito[["ID", "Text"]].merge(df_previsto[["ID", *CATEGORIAS]], on="ID")
            st.dataframe(tabela_modelo, width="stretch", hide_index=True)


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
    st.altair_chart(grafico, width="stretch")
    sem_positivos = categorias_sem_positivos(df_metricas)
    if sem_positivos:
        if metrica_escolhida in METRICAS_SEM_POSITIVOS_INDEFINIDAS:
            st.caption(
                f"{', '.join(sem_positivos)} (N/A): não tem nenhum positivo no gabarito, então esta métrica não está "
                "definida e não há barra. Veja a nota abaixo da tabela detalhada."
            )
        else:
            st.caption(
                f"Atenção: a acurácia de {', '.join(sem_positivos)} é alta só porque os modelos marcam 0 quase sempre; "
                "a categoria não tem nenhum positivo no gabarito, então ela não mede capacidade de identificá-la."
            )

    secao_explicacoes()
    secao_tabelas(df_metricas, modelos_disponiveis)
    secao_intervalos_confianca(modelos_disponiveis, cores_disponiveis, usar_teste)
    secao_qualidade_respostas(respostas)
    secao_consistencia()
    secao_respostas_por_modelo(df_gabarito, respostas)


if __name__ == "__main__":
    main()
