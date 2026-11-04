# feedback-formativo

## Finalidade

Projeto do TCC que compara como três IAs (Claude, Gemini e OpenAI) classificam
feedbacks educacionais nas 7 boas práticas de feedback de Nicol & Macfarlane-Dick
(GP1–GP7). Cada feedback do dataset já tem uma classificação feita por
especialistas (o gabarito humano: dois anotadores e, nos casos de divergência,
mais dois especialistas). O projeto pede a mesma classificação para cada IA e
mede o quanto ela concorda com o gabarito, categoria por categoria, com
**acurácia, precisão, recall, F1, MCC (coeficiente de correlação de Matthews) e
Kappa de Cohen**, com intervalos de confiança de 95% por bootstrap.

O dataset também traz as colunas FT, FP, FR e FS (níveis de Hattie & Timperley),
que **não são usadas**: o trabalho avalia só as boas práticas.

## Estrutura

| Caminho | O que é |
|---------|---------|
| `src/claude_client.py`, `gemini_client.py`, `openai_client.py` | Um wrapper por provedor, todos com a mesma função `gerar_resposta(prompt)` |
| `src/config.py` | Lê e valida as chaves de API do `.env` |
| `src/modelos.py` | Registro central dos 3 modelos (id, nome de exibição, cor no gráfico) |
| `src/prompts.py` | Prompt de classificação (versão original e versão revisada) e o parser que extrai GP1–GP7 da resposta |
| `src/metricas.py` | Acurácia, precisão, recall, F1, MCC e Kappa por categoria, comparando previsões com o gabarito |
| `src/intervalos_confianca.py` | IC 95% por bootstrap de F1, MCC e Kappa, e IC 95% pareado da diferença entre duas classificações |
| `scripts/gerar_respostas.py` | Chama os modelos e salva as respostas (retoma de onde parou; `--plano` mostra o custo sem chamar a API) |
| `scripts/calcular_ic95.py` | IC 95% de cada modelo a partir dos CSVs existentes (sem API) |
| `scripts/calcular_ic95_diferencas.py` | IC 95% da diferença entre cada par de modelos (sem API) |
| `scripts/teste_consistencia.py` | Mede a variação do Claude entre duas execuções (por padrão só mostra o plano e o custo) |
| `scripts/dividir_dados.py` | Divide os feedbacks em desenvolvimento (200) e teste (800) para o experimento complementar |
| `scripts/comparar_prompts.py` | Compara o prompt original com o revisado no conjunto de teste (sem API) |
| `app.py` | App Streamlit com todas as comparações |
| `docs/execucao.md` | Como as respostas foram geradas (modelos, temperatura, JSON, datas, origem do prompt, estatística) |
| `docs/prompt_revisado.md` | Experimento complementar do prompt revisado |
| `data/raw/dataset.csv` | Dataset com o gabarito humano (1000 feedbacks) |
| `data/splits/` | IDs de desenvolvimento e de teste (saída de `dividir_dados.py`) |
| `data/processed/respostas_<modelo>.csv` | Respostas do experimento principal |
| `data/processed/metricas_ic95.csv`, `diferencas_ic95.csv` | Intervalos de confiança (saídas dos scripts de IC) |

## Como rodar

```bash
cd feedback-formativo
source .venv/Scripts/activate   # Git Bash
pip install -r requirements.txt
```

Copie `.env.example` para `.env` e preencha as chaves dos provedores que for usar:

```
OPENAI_API_KEY=...
GEMINI_API_KEY=...
CLAUDE_API_KEY=...
```

### 1. Gerar as respostas dos modelos (experimento principal)

```bash
python scripts/gerar_respostas.py
```

Isso chama cada modelo configurado (basta ter a chave correspondente no `.env`)
para **as 1000 linhas do dataset** e salva um CSV por modelo em
`data/processed/respostas_<modelo>.csv`.

⚠️ Isso faz ~1000 chamadas de API por modelo (1 por segundo, de propósito,
para não estourar limite de requisições) e consome créditos de verdade. Se o
script for interrompido, é só rodar de novo: ele pula os IDs que já estão
salvos no CSV e continua de onde parou.

**Teste rápido e barato** (útil pra conferir o prompt antes de rodar tudo):

```bash
python scripts/gerar_respostas.py --n 10
```

Processa só as 10 primeiras linhas e salva em
`data/processed/respostas_<modelo>_teste.csv`, sem mexer nos resultados da
rodada completa. Toda vez que rodar com `--n`, começa do zero.

### 2. Calcular os intervalos de confiança (sem custo)

```bash
python scripts/calcular_ic95.py              # IC 95% de cada modelo → metricas_ic95.csv
python scripts/calcular_ic95_diferencas.py   # IC 95% da diferença entre modelos → diferencas_ic95.csv
```

Bootstrap com 1000 reamostragens e semente fixa; não chamam nenhuma API. O
segundo responde "um modelo é melhor que o outro?": se o intervalo da
diferença contém 0, não há evidência de diferença.

### 3. Abrir o app de comparação

```bash
streamlit run app.py
```

Abre no navegador (`http://localhost:8501`) com:
- o gráfico comparativo por categoria;
- o resumo geral;
- a tabela de métricas detalhadas, com o número de positivos no gabarito;
- os intervalos de confiança e a diferença entre modelos;
- a contagem de falhas de parsing;
- o teste de consistência do Claude;
- a comparação dos prompts, se já tiver sido rodada;
- as respostas de cada modelo.

A caixinha **"Usar resultados de teste"** na barra lateral mostra os `*_teste.csv` (gerados com `--n`).

O **resumo geral** é a média das categorias que têm positivos no gabarito
(GP1–GP6). GP7 não tem nenhum positivo e aparece como **N/A** nas métricas que
dependem de positivos.

### 4. Teste de consistência do Claude (opcional, custa dinheiro)

O Claude roda sem `temperature=0`, então pode responder diferente da mesma
pergunta. Para medir isso:

```bash
python scripts/teste_consistencia.py              # só mostra o plano e o custo estimado (não chama a API)
python scripts/teste_consistencia.py --executar   # faz ~100 chamadas pagas ao Claude
python scripts/teste_consistencia.py --comparar   # compara um CSV já gerado (sem API)
```

Sorteia 100 feedbacks (semente fixa), pede ao Claude de novo e compara com
`respostas_claude.csv` (% de concordância e Kappa por GP). A rodada original
nunca é alterada: a nova vai para `data/processed/consistencia_claude.csv`.

### 5. Experimento complementar: prompt revisado (exploratório)

O experimento principal continua sendo o **prompt original nos 1000 feedbacks**.
Este experimento testa uma versão do prompt em que só GP1 e GP6 foram
reescritas para seguir a forma como os anotadores aplicaram o guia. Os
detalhes estão em `docs/prompt_revisado.md`.

```bash
python scripts/dividir_dados.py      # divisão 200/800, feita uma única vez (já existe em data/splits/)
python scripts/gerar_respostas.py --prompt revisado --ids data/splits/teste_ids.csv --plano            # custo, sem API
python scripts/gerar_respostas.py --prompt revisado --ids data/splits/teste_ids.csv --modelo gemini    # um modelo por vez
python scripts/comparar_prompts.py   # sem API → data/processed/comparacao_prompts.csv
```

O prompt revisado foi escrito olhando só os 200 feedbacks de desenvolvimento,
e a comparação é feita só nos 800 de teste. O prompt original não precisa rodar
de novo, porque as respostas dele nesses 800 já existem.
