"""
Coleta classificações dos modelos (Claude, Gemini, OpenAI) para as categorias
GP1-GP7 e salva os resultados em data/processed/.

Se um resultado parcial já existir (de uma execução anterior ou interrompida),
os IDs já processados são pulados — só continua de onde parou.

Uso (experimento principal, prompt original):
    python scripts/gerar_respostas.py            # dataset completo
    python scripts/gerar_respostas.py --n 10      # só as 10 primeiras linhas,
                                                   # salva em respostas_<modelo>_teste.csv
                                                   # (não mexe nos resultados da rodada completa)

Uso (experimento complementar, prompt revisado, só no conjunto de teste):
    python scripts/gerar_respostas.py --prompt revisado --ids data/splits/teste_ids.csv --plano
    python scripts/gerar_respostas.py --prompt revisado --ids data/splits/teste_ids.csv --modelo gemini
    → salva em respostas_<modelo>_revisado.csv

--plano só mostra quantas chamadas seriam feitas, o custo estimado e o arquivo
de saída: NÃO chama nenhuma API.
"""

import argparse
import importlib
import sys
import time
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.append(str(RAIZ))

from src.modelos import MODELOS
from src.prompts import CATEGORIAS, VERSOES_PROMPT, montar_prompt, parse_resposta

CAMINHO_DATASET = RAIZ / "data" / "raw" / "dataset.csv"
PASTA_SAIDA = RAIZ / "data" / "processed"
TENTATIVAS_POR_LINHA = 3  # tentativas antes de desistir de uma linha (erros de rede/rate limit são passageiros)

# --- Estimativa de custo do --plano (só informativa) -----------------------
# Preços de referência em US$ por milhão de tokens (entrada, saída). CONFIRA nas
# páginas de preço de cada empresa antes de rodar; os valores mudam.
PRECOS_REFERENCIA = {
    "claude": (2.00, 10.00),   # claude-sonnet-5 (mesmo valor usado em teste_consistencia.py)
    "gemini": (0.10, 0.40),    # gemini-3.1-flash-lite (estimativa; confira)
    "openai": (0.15, 0.60),    # gpt-4o-mini
}
CARACTERES_POR_TOKEN = 3.3  # português; valor baixo de propósito (superestima tokens)
TOKENS_SAIDA_POR_CHAMADA = 50  # o JSON de resposta tem ~35-45 tokens
SEGUNDOS_POR_CHAMADA = 4  # ~3 s de resposta + 1 s de pausa entre chamadas


def carregar_modelos(ids_modelos: list[str]) -> dict:
    """Importa o cliente de cada modelo separadamente, para permitir rodar só
    com as chaves de API que estiverem configuradas no .env."""
    modelos = {}
    for id_modelo in ids_modelos:
        info = MODELOS[id_modelo]
        try:
            modelos[id_modelo] = importlib.import_module(f"src.{id_modelo}_client")
        except ValueError as erro:
            print(f"[aviso] {info['nome']} não será avaliado: {erro}")
    return modelos


def avaliar_modelo(
    nome_modelo: str, cliente, df: pd.DataFrame, caminho_saida: Path, versao_prompt: str = "original"
) -> None:
    ja_processados = set()
    if caminho_saida.exists():
        ja_processados = set(pd.read_csv(caminho_saida)["ID"])

    pendentes = df[~df["ID"].isin(ja_processados)]
    if ja_processados:
        print(f"  [{nome_modelo}] retomando: {len(ja_processados)} já processados, faltam {len(pendentes)}")
    if pendentes.empty:
        return

    for i, row in enumerate(pendentes.itertuples(), start=1):
        prompt = montar_prompt(row.Text, versao_prompt)
        resposta = ""
        previsao = {categoria: None for categoria in CATEGORIAS}

        for tentativa in range(1, TENTATIVAS_POR_LINHA + 1):
            try:
                resposta = cliente.gerar_resposta(prompt)
                previsao = parse_resposta(resposta)
                break
            except Exception as erro:
                if tentativa < TENTATIVAS_POR_LINHA:
                    print(f"  [{nome_modelo}] erro no ID {row.ID} (tentativa {tentativa}/{TENTATIVAS_POR_LINHA}): {erro} — tentando de novo...")
                    time.sleep(5)
                else:
                    print(f"  [{nome_modelo}] erro no ID {row.ID}, desistindo após {TENTATIVAS_POR_LINHA} tentativas: {erro}")

        linha = pd.DataFrame([{"ID": row.ID, "resposta_bruta": resposta, **previsao}])
        linha.to_csv(caminho_saida, mode="a", header=not caminho_saida.exists(), index=False)

        print(f"  [{nome_modelo}] ID {row.ID} processado ({i}/{len(pendentes)})")
        time.sleep(1)  # evita estourar limite de requisições por segundo


def mostrar_plano(id_modelo: str, df: pd.DataFrame, caminho_saida: Path, versao_prompt: str) -> None:
    """Mostra o que seria feito para um modelo, sem chamar a API."""
    ja_processados = set(pd.read_csv(caminho_saida)["ID"]) if caminho_saida.exists() else set()
    pendentes = df[~df["ID"].isin(ja_processados)]
    caracteres = sum(len(montar_prompt(texto, versao_prompt)) for texto in pendentes["Text"])
    tokens_entrada = caracteres / CARACTERES_POR_TOKEN
    tokens_saida = len(pendentes) * TOKENS_SAIDA_POR_CHAMADA
    preco_entrada, preco_saida = PRECOS_REFERENCIA[id_modelo]
    custo = tokens_entrada / 1e6 * preco_entrada + tokens_saida / 1e6 * preco_saida
    print(f"\n[{MODELOS[id_modelo]['nome']}] prompt {versao_prompt}")
    print(f"  Feedbacks: {len(df)} (já processados: {len(ja_processados & set(df['ID']))}, faltam {len(pendentes)})")
    print(f"  Chamadas à API: {len(pendentes)} (até {len(pendentes) * TENTATIVAS_POR_LINHA} no pior caso, se a API falhar)")
    print(f"  Tokens estimados: ~{tokens_entrada:,.0f} de entrada + ~{tokens_saida:,.0f} de saída")
    print(f"  Custo estimado: ~US$ {custo:.2f} (pior caso: ~US$ {custo * TENTATIVAS_POR_LINHA:.2f}; confira os preços)")
    print(f"  Tempo estimado: ~{len(pendentes) * SEGUNDOS_POR_CHAMADA / 60:.0f} minutos")
    print(f"  Saída: {caminho_saida}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n", type=int, default=None,
        help="Processa só as N primeiras linhas do dataset (teste rápido/barato); "
             "salva em respostas_<modelo>_teste.csv, sem tocar na rodada completa",
    )
    parser.add_argument(
        "--prompt", choices=list(VERSOES_PROMPT), default="original",
        help="Versão do prompt (padrão: original). Com 'revisado', salva em respostas_<modelo>_revisado.csv",
    )
    parser.add_argument(
        "--ids", type=Path, default=None,
        help="CSV com uma coluna ID: processa só esses feedbacks (ex.: data/splits/teste_ids.csv)",
    )
    parser.add_argument(
        "--modelo", choices=list(MODELOS), default=None,
        help="Roda só este modelo (padrão: todos os que tiverem chave no .env)",
    )
    parser.add_argument(
        "--plano", action="store_true",
        help="Só mostra o número de chamadas, o custo estimado e o arquivo de saída. NÃO chama a API.",
    )
    args = parser.parse_args()

    if args.n and args.prompt != "original":
        parser.error("--n é só para testar o prompt original; para o revisado use --ids")

    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CAMINHO_DATASET)

    if args.ids:
        ids = set(pd.read_csv(args.ids)["ID"])
        df = df[df["ID"].isin(ids)]
        if len(df) != len(ids):
            parser.error(f"{len(ids) - len(df)} IDs de {args.ids} não existem no dataset")

    sufixo = ""
    if args.n:
        df = df.head(args.n)
        sufixo = "_teste"
    elif args.prompt != "original":
        sufixo = f"_{args.prompt}"

    if args.prompt == "original" and args.ids and not args.n:
        parser.error("com o prompt original, --ids sobrescreveria a rodada completa; os resultados já existem")

    ids_modelos = [args.modelo] if args.modelo else list(MODELOS)

    if args.plano:
        for id_modelo in ids_modelos:
            mostrar_plano(id_modelo, df, PASTA_SAIDA / f"respostas_{id_modelo}{sufixo}.csv", args.prompt)
        print("\nNada foi enviado à API. Para rodar de verdade, repita o comando sem --plano.")
        return

    modelos = carregar_modelos(ids_modelos)
    if not modelos:
        print("Nenhuma chave de API configurada em .env. Configure ao menos uma e tente novamente.")
        return

    for nome_modelo, cliente in modelos.items():
        caminho_saida = PASTA_SAIDA / f"respostas_{nome_modelo}{sufixo}.csv"
        if args.n and caminho_saida.exists():
            caminho_saida.unlink()  # teste sempre roda do zero, pra refletir o prompt atual

        if args.n:
            descricao = f"{len(df)} feedbacks de teste"
        elif args.ids:
            descricao = f"{len(df)} feedbacks de {args.ids.name}, prompt {args.prompt}"
        else:
            descricao = f"{len(df)} feedbacks (dataset completo)"
        print(f"\nAvaliando {nome_modelo} ({descricao})...")
        avaliar_modelo(nome_modelo, cliente, df, caminho_saida, args.prompt)
        print(f"  -> resultados em {caminho_saida}")


if __name__ == "__main__":
    main()
