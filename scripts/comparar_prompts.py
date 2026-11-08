import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.append(str(RAIZ))

from src.intervalos_confianca import N_BOOTSTRAP, SEMENTE, diferenca_pareada
from src.metricas import resumir_falhas
from src.modelos import MODELOS
from src.prompts import CATEGORIAS

CAMINHO_DATASET = RAIZ / "data" / "raw" / "dataset.csv"
CAMINHO_TESTE = RAIZ / "data" / "splits" / "teste_ids.csv"
PASTA_PROCESSADOS = RAIZ / "data" / "processed"
CAMINHO_SAIDA = PASTA_PROCESSADOS / "comparacao_prompts.csv"


def main():
    if not CAMINHO_TESTE.exists():
        print(f"{CAMINHO_TESTE} não existe. Rode primeiro: python scripts/dividir_dados.py")
        return

    ids_teste = set(pd.read_csv(CAMINHO_TESTE)["ID"])
    df_gabarito = pd.read_csv(CAMINHO_DATASET)
    df_gabarito = df_gabarito[df_gabarito["ID"].isin(ids_teste)]
    categorias = [c for c in CATEGORIAS if df_gabarito[c].sum() > 0]

    resultados = []
    for id_modelo, info in MODELOS.items():
        caminho_original = PASTA_PROCESSADOS / f"respostas_{id_modelo}.csv"
        caminho_revisado = PASTA_PROCESSADOS / f"respostas_{id_modelo}_revisado.csv"
        if not caminho_revisado.exists():
            print(f"[{info['nome']}] sem {caminho_revisado.name}: ainda não rodado com o prompt revisado.")
            continue

        original = pd.read_csv(caminho_original)
        revisado = pd.read_csv(caminho_revisado)
        faltando = ids_teste - set(revisado["ID"])
        falhas = resumir_falhas(revisado)
        if faltando:
            print(f"[{info['nome']}] atenção: {len(faltando)} IDs de teste ainda sem resposta revisada (rodada incompleta).")
        if falhas["com_gp_nulo"]:
            print(f"[{info['nome']}] atenção: {falhas['com_gp_nulo']} respostas revisadas com algum GP sem valor (ficam de fora).")

        print(f"[{info['nome']}] revisado − original ({N_BOOTSTRAP} reamostragens pareadas, semente {SEMENTE})...")
        df = diferenca_pareada(df_gabarito, revisado, original, categorias, "Revisado", "Original")
        df = df.rename(columns={"valor_a": "valor_revisado", "valor_b": "valor_original"})
        df.insert(0, "modelo", info["nome"])
        resultados.append(df)

    if not resultados:
        print("\nNenhum respostas_<modelo>_revisado.csv encontrado. Nada a comparar ainda.")
        return

    df_final = pd.concat(resultados, ignore_index=True)
    df_final.to_csv(CAMINHO_SAIDA, index=False)
    print(f"\n  -> {len(df_final)} linhas salvas em {CAMINHO_SAIDA}\n")

    for _, linha in df_final[df_final["metrica"] == "kappa"].iterrows():
        print(
            f"{linha['modelo']:7} {linha['categoria']:6} Kappa original {linha['valor_original']:.2f} → revisado "
            f"{linha['valor_revisado']:.2f} | dif. {linha['diferenca']:+.2f} "
            f"[{linha['ic95_inferior']:+.2f}; {linha['ic95_superior']:+.2f}] → {linha['conclusao']}"
        )


if __name__ == "__main__":
    main()
