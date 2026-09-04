# feedback-formativo

## Finalidade

Projeto que compara como três IAs (Claude, Gemini e OpenAI) classificam
feedbacks educacionais nas 7 categorias de boas práticas de feedback formativo
(GP1-GP7, baseadas em Nicol & Macfarlane-Dick). Cada feedback do dataset já
tem uma classificação feita por um professor (o gabarito humano); o projeto
pede a mesma classificação para cada IA e mede o quanto elas concordam com o
professor (acurácia, precisão, recall e F1, categoria por categoria), para
comparar o desempenho dos três modelos nessa tarefa.

## Estrutura

| Caminho | O que é |
|---------|---------|
| `src/claude_client.py`, `gemini_client.py`, `openai_client.py` | Um wrapper por provedor, todos com a mesma função `gerar_resposta(prompt)` |
| `src/config.py` | Lê e valida as chaves de API do `.env` |
| `src/modelos.py` | Registro central dos 3 modelos (id, nome de exibição, cor no gráfico) |
| `src/prompts.py` | Prompt de classificação e o parser que extrai GP1-GP7 da resposta do modelo |
| `src/metricas.py` | Cálculo de acurácia/precisão/recall/F1 comparando previsões com o gabarito |
| `scripts/gerar_respostas.py` | Chama os 3 modelos e salva os resultados (retoma de onde parou se for interrompido) |
| `app.py` | App Streamlit que mostra as métricas de comparação entre os modelos |
| `data/raw/dataset.csv` | Dataset com o gabarito humano (1000 feedbacks) |
| `data/processed/respostas_*.csv` | Saída de `gerar_respostas.py` (gerado localmente, não versionado) |

## Como rodar

```bash
cd feedback-formativo
source .venv/Scripts/activate   # Git Bash
pip install -r requirements.txt
```

Configure `.env` (veja `.env.example`) com as chaves dos provedores que for usar:

```
OPENAI_API_KEY=...
GEMINI_API_KEY=...
CLAUDE_API_KEY=...
```

### 1. Gerar as respostas dos modelos

```bash
python scripts/gerar_respostas.py
```

Isso chama cada modelo configurado (basta ter a chave correspondente no `.env`)
para **as 1000 linhas do dataset** e salva um CSV por modelo em
`data/processed/respostas_<modelo>.csv`.

⚠️ Isso faz ~1000 chamadas de API por modelo (1 por segundo, de propósito,
para não estourar limite de requisições) — leva bem mais de 15 minutos por
modelo e consome créditos de verdade nas três contas de API. Se o script for
interrompido, é só rodar de novo: ele pula os IDs que já estão salvos no CSV e
continua de onde parou.

**Teste rápido e barato** (útil pra conferir o prompt antes de rodar tudo):

```bash
python scripts/gerar_respostas.py --n 10
```

Processa só as 10 primeiras linhas e salva em
`data/processed/respostas_<modelo>_teste.csv` — não mexe nos resultados da
rodada completa. Toda vez que rodar com `--n`, começa do zero (não retoma),
pra sempre refletir o prompt mais recente.

### 2. Abrir o app de comparação

```bash
streamlit run app.py
```

Abre no navegador (`http://localhost:8501`) com o gráfico comparativo por
categoria, a tabela de métricas detalhadas e as respostas de cada modelo no
formato do dataset original. Por padrão mostra os resultados da rodada
completa; a caixinha **"Usar resultados de teste"** na barra lateral troca
para mostrar os `*_teste.csv` (gerados com `--n`).
