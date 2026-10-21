"""
Teste de consistência do Claude (claude-sonnet-5).

O Claude roda SEM temperature=0 (a API não aceita o parâmetro para esse modelo),
então a mesma pergunta pode receber respostas diferentes em execuções diferentes.
Este script mede o quanto: sorteia uma amostra fixa de feedbacks, pede ao Claude
de novo SÓ para eles e compara com a rodada original (respostas_claude.csv).

Não altera nada que já existe: a nova rodada vai para
data/processed/consistencia_claude.csv e o resumo para
data/processed/consistencia_claude_resumo.csv.

Três modos (por padrão NÃO chama a API):
    python scripts/teste_consistencia.py              # só mostra o plano e o custo estimado
    python scripts/teste_consistencia.py --executar   # FAZ as chamadas pagas ao Claude
    python scripts/teste_consistencia.py --comparar   # só compara um CSV já gerado (sem API)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.append(str(RAIZ))
sys.path.append(str(RAIZ / "scripts"))

from src.metricas import comparar_execucoes
from src.prompts import montar_prompt

CAMINHO_DATASET = RAIZ / "data" / "raw" / "dataset.csv"
PASTA_PROCESSADOS = RAIZ / "data" / "processed"
CAMINHO_ORIGINAL = PASTA_PROCESSADOS / "respostas_claude.csv"
CAMINHO_NOVA = PASTA_PROCESSADOS / "consistencia_claude.csv"
CAMINHO_RESUMO = PASTA_PROCESSADOS / "consistencia_claude_resumo.csv"

N_AMOSTRA = 100
SEMENTE = 42

# --- Estimativa de custo (só informativa) ---------------------------------
# Preço de referência do claude-sonnet-5 em US$ por milhão de tokens. Confira
# na página de preços da Anthropic antes de decidir.
PRECO_ENTRADA_POR_MTOK = 2.00
PRECO_SAIDA_POR_MTOK = 10.00
CARACTERES_POR_TOKEN = 3.3  # português; valor baixo de propósito (superestima tokens)
TOKENS_SAIDA_POR_CHAMADA = 50  # o JSON de resposta tem ~35-45 tokens; sobra um pouco
TENTATIVAS_MAXIMAS = 3  # gerar_respostas.py tenta até 3 vezes se a API falhar
SEGUNDOS_POR_CHAMADA = 4  # ~3 s de resposta + 1 s de pausa entre chamadas


def sortear_ids(df_dataset: pd.DataFrame) -> list[int]:
    """Os mesmos N_AMOSTRA IDs em qualquer execução (semente fixa)."""
    rng = np.random.default_rng(SEMENTE)
    return sorted(int(i) for i in rng.choice(df_dataset["ID"].to_numpy(), size=N_AMOSTRA, replace=False))


def estimar_custo(df_amostra: pd.DataFrame) -> dict[str, float]:
    caracteres = sum(len(montar_prompt(texto)) for texto in df_amostra["Text"])
    tokens_entrada = caracteres / CARACTERES_POR_TOKEN
    tokens_saida = len(df_amostra) * TOKENS_SAIDA_POR_CHAMADA

    def custo(entrada: float, saida: float) -> float:
        return entrada / 1e6 * PRECO_ENTRADA_POR_MTOK + saida / 1e6 * PRECO_SAIDA_POR_MTOK

    return {
        "chamadas": len(df_amostra),
        "tokens_entrada": tokens_entrada,
        "tokens_saida": tokens_saida,
        "custo_usd": custo(tokens_entrada, tokens_saida),
        "custo_usd_pior_caso": custo(tokens_entrada, tokens_saida) * TENTATIVAS_MAXIMAS,
        "minutos": len(df_amostra) * SEGUNDOS_POR_CHAMADA / 60,
    }


def mostrar_plano(df_amostra: pd.DataFrame) -> None:
    est = estimar_custo(df_amostra)
    primeiros_ids = sorted(df_amostra["ID"])[:8]
    print(f"Amostra: {N_AMOSTRA} feedbacks sorteados com semente {SEMENTE} (primeiros IDs: {primeiros_ids}...)")
    print(f"Chamadas à API do Claude: {est['chamadas']} (uma por feedback; até {est['chamadas'] * TENTATIVAS_MAXIMAS} no pior caso, se a API falhar)")
    print(f"Tokens estimados: ~{est['tokens_entrada']:,.0f} de entrada + ~{est['tokens_saida']:,.0f} de saída")
    print(f"Custo estimado: ~US$ {est['custo_usd']:.2f} (pior caso com todas as tentativas: ~US$ {est['custo_usd_pior_caso']:.2f})")
    print(f"Tempo estimado: ~{est['minutos']:.0f} minutos")
    print(f"Saída: {CAMINHO_NOVA}  (o arquivo da rodada original NÃO é tocado)")
    print("\nNada foi enviado à API. Para rodar de verdade: python scripts/teste_consistencia.py --executar")


def executar(df_amostra: pd.DataFrame) -> None:
    # Importados só aqui: criar o cliente do Claude exige a CLAUDE_API_KEY e este
    # é o único modo que realmente chama a API.
    from gerar_respostas import avaliar_modelo
    from src import claude_client

    assert CAMINHO_NOVA != CAMINHO_ORIGINAL, "a nova rodada nunca pode sobrescrever a original"
    # avaliar_modelo é o mesmo laço de gerar_respostas.py (mesmo prompt, mesmas
    # tentativas) e retoma de onde parou se o arquivo já existir.
    avaliar_modelo("claude (consistência)", claude_client, df_amostra, CAMINHO_NOVA)


def comparar() -> None:
    if not CAMINHO_NOVA.exists():
        print(f"{CAMINHO_NOVA.name} ainda não existe. Rode com --executar primeiro.")
        return
    resumo = comparar_execucoes(pd.read_csv(CAMINHO_ORIGINAL), pd.read_csv(CAMINHO_NOVA))
    resumo.to_csv(CAMINHO_RESUMO, index=False)

    exibicao = resumo.copy()
    exibicao["pct_concordancia"] = exibicao["pct_concordancia"].map("{:.1f}%".format)
    exibicao["kappa"] = exibicao["kappa"].map(lambda k: "N/A" if pd.isna(k) else f"{k:.2f}")
    print(exibicao.fillna("-").to_string(index=False))
    print("\nKappa N/A = indefinido (as duas execuções marcaram sempre o mesmo valor, ex.: só 0).")
    print(f"Resumo salvo em {CAMINHO_RESUMO}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument("--executar", action="store_true", help="faz as chamadas pagas à API do Claude e compara")
    grupo.add_argument("--comparar", action="store_true", help="só compara um CSV já gerado (não chama a API)")
    args = parser.parse_args()

    df_dataset = pd.read_csv(CAMINHO_DATASET)
    ids = sortear_ids(df_dataset)
    df_amostra = df_dataset[df_dataset["ID"].isin(ids)]

    if args.comparar:
        comparar()
    elif args.executar:
        executar(df_amostra)
        comparar()
    else:
        mostrar_plano(df_amostra)


if __name__ == "__main__":
    main()
