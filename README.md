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
| `src/prompts.py` | Prompt de classificação e o parser que extrai GP1-GP7 da resposta do modelo |
| `src/metricas.py` | Cálculo de acurácia/precisão/recall/F1 comparando previsões com o gabarito |
| `src/intervalos_confianca.py` | Intervalo de confiança de 95% (bootstrap) para F1, MCC e Kappa, de cada modelo e da diferença entre pares de modelos |
| `scripts/gerar_respostas.py` | Chama os 3 modelos e salva os resultados (retoma de onde parou se for interrompido) |
| `scripts/calcular_ic95.py` | Calcula os intervalos de confiança a partir dos CSVs existentes (sem API) |
| `scripts/teste_consistencia.py` | Mede a variação do Claude entre duas execuções (por padrão só mostra o plano e o custo) |
| `scripts/dividir_dados.py` | Divide os feedbacks em desenvolvimento (200) e teste (800) para o experimento complementar |
| `scripts/comparar_prompts.py` | Compara o prompt original com o revisado no conjunto de teste (sem API) |
| `app.py` | App Streamlit com todas as comparações |
| `docs/execucao.md` | Como as respostas foram geradas (modelos, temperatura, JSON, datas, origem do prompt, estatística) |
| `docs/prompt_revisado.md` | Experimento complementar do prompt revisado |
| `data/raw/dataset.csv` | Dataset com o gabarito humano (1000 feedbacks) |
<<<<<<< HEAD
| `data/processed/respostas_*.csv` | Saída de `gerar_respostas.py` |
| `data/processed/metricas_ic95.csv` | Saída de `calcular_ic95.py` (estimativa e IC 95% por modelo, categoria e métrica) |
| `data/processed/diferencas_ic95.csv` | Saída de `calcular_ic95.py` (diferença A − B e IC 95% por par de modelos, categoria e métrica) |
=======
| `data/splits/` | IDs de desenvolvimento e de teste (saída de `dividir_dados.py`) |
| `data/processed/respostas_<modelo>.csv` | Respostas do experimento principal |
| `data/processed/metricas_ic95.csv`, `diferencas_ic95.csv` | Intervalos de confiança (saídas dos scripts de IC) |
>>>>>>> 130aa4bddc1abb2ed123e6fcd710bf1f9e6ff2c5

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

<<<<<<< HEAD
Lê os CSVs que já existem e grava `data/processed/metricas_ic95.csv` (IC de
cada modelo) e `data/processed/diferencas_ic95.csv` (IC da diferença entre
cada par de modelos, com bootstrap pareado). Usa 1000 reamostragens e semente
fixa, e não chama nenhuma API. O app mostra esses arquivos se eles existirem.
=======
Bootstrap com 1000 reamostragens e semente fixa; não chamam nenhuma API. O
segundo responde "um modelo é melhor que o outro?": se o intervalo da
diferença contém 0, não há evidência de diferença.
>>>>>>> 130aa4bddc1abb2ed123e6fcd710bf1f9e6ff2c5

### 3. Abrir o app de comparação

```bash
streamlit run app.py
```

<<<<<<< HEAD
Abre no navegador (`http://localhost:8501`) com o gráfico comparativo por
categoria, o resumo geral, a tabela de métricas detalhadas (com o número de
positivos no gabarito), os intervalos de confiança, o intervalo da diferença
entre os modelos (para dizer se um é significativamente melhor), a contagem de falhas de
parsing, o resultado do teste de consistência do Claude (se já tiver sido
rodado) e as respostas de cada modelo no formato do dataset original. Por
padrão mostra os resultados da rodada completa; a caixinha **"Usar resultados
de teste"** na barra lateral troca para mostrar os `*_teste.csv` (gerados com
`--n`).
=======
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
>>>>>>> 130aa4bddc1abb2ed123e6fcd710bf1f9e6ff2c5

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

## Como o código funciona (detalhes de implementação)

O código não tem comentários; as decisões de implementação estão registradas aqui.

### Clientes dos modelos (`src/*_client.py`, `src/config.py`, `src/modelos.py`)

- Cada cliente expõe `gerar_resposta(prompt)` e lê a chave do `.env` por `obter_chave_api`, que
  lança erro se a chave não estiver configurada.
- **Claude** (`claude-sonnet-5`): `max_tokens=1024`, sem `temperature` (a API rejeita o parâmetro
  para esse modelo, com o erro "deprecated for this model") e sem JSON forçado.
- **Gemini** (`gemini-3.1-flash-lite`) e **OpenAI** (`gpt-4o-mini`): `temperature=0`, porque numa
  tarefa de classificação queremos a resposta mais provável, e saída forçada em JSON válido.
- Em `src/modelos.py`, a chave de cada modelo (`claude`, `gemini`, `openai`) é usada no nome do
  módulo do cliente (`src/<id>_client.py`) e no nome do arquivo de resultados
  (`data/processed/respostas_<id>.csv`).

### Prompt e leitura da resposta (`src/prompts.py`)

- `montar_prompt(texto, versao="original")` monta o prompt com a definição, um exemplo positivo e
  um negativo de cada GP. `versao="revisado"` usa `DESCRICOES_CATEGORIAS_REVISADO`, que muda só
  GP1 e GP6 (veja `docs/prompt_revisado.md`). Os exemplos do revisado são feedbacks do conjunto
  de desenvolvimento (IDs em `IDS_EXEMPLOS_REVISADO`).
- `parse_resposta` remove cercas de markdown (` ```json `), tenta ler como JSON e, se falhar, usa
  expressão regular no texto bruto. Quando uma categoria não vem no formato 0/1, ela fica `None`.

### Coleta das respostas (`scripts/gerar_respostas.py`)

- Os clientes são importados um a um, para rodar só com as chaves que estiverem no `.env`.
- Uma chamada por feedback, com pausa de 1 s entre chamadas (limite de requisições) e até 3
  tentativas (5 s de espera) em caso de erro de API. Resposta mal formatada não gera nova tentativa.
- Retoma de onde parou: IDs que já estão no CSV de saída são pulados. Com `--n`, o arquivo de teste
  é apagado antes, para sempre refletir o prompt atual.
- `--plano` estima o custo com preços de referência por milhão de tokens (Claude US$ 2/10, Gemini
  US$ 0,10/0,40, OpenAI US$ 0,15/0,60 de entrada/saída; **confira os preços antes de rodar**),
  3,3 caracteres por token (valor baixo de propósito, para não subestimar), 50 tokens de saída
  por chamada e cerca de 4 s por chamada.
- Com o prompt original, `--ids` é recusado, para não sobrescrever a rodada completa.

### Métricas (`src/metricas.py`)

- `calcular_metricas` compara as respostas com o gabarito, por categoria, com o scikit-learn.
  Linhas com GP sem valor (`None`) ficam de fora só naquela categoria.
- Convenções para casos indefinidos: precisão, recall e F1 valem 0 (`zero_division=0`); MCC e
  Kappa, que ficam 0/0 quando real e previsto são constantes, também valem 0.
- `marcar_nao_definidas` troca precisão, recall, F1, MCC e Kappa por N/A nas categorias sem nenhum
  positivo no gabarito (GP7). Serve só para a apresentação, para não confundir "não dá para medir"
  com "o modelo errou tudo". A acurácia continua definida, mas só reflete o quanto o modelo marcou 0.
- `media_por_modelo` é a média simples das categorias com positivos no gabarito (GP1–GP6).
- `resumir_falhas` conta respostas vazias (a API falhou em todas as tentativas) e respostas com
  algum GP sem valor.
- `comparar_execucoes` (teste de consistência) dá, por GP, a % de concordância e o Kappa entre
  duas execuções do mesmo modelo, e a % de feedbacks com os 7 GP idênticos. Kappa indefinido
  (as duas execuções só marcaram 0) fica N/A.

### Intervalos de confiança (`src/intervalos_confianca.py`)

- **IC de cada modelo** (`calcular_ic95`): bootstrap percentil. Sorteia os feedbacks com reposição
  1000 vezes, recalcula F1, MCC e Kappa e usa os percentis 2,5% e 97,5%. Para ser rápido, as
  métricas de cada reamostragem são calculadas com numpy a partir das contagens VP/FP/FN/VN, com
  as mesmas fórmulas e convenções do scikit-learn. A semente depende só da categoria
  (`default_rng([42, índice])`), então todos os modelos usam as mesmas reamostragens. Categorias
  sem positivos (GP7) ficam vazias, com uma observação.
- **IC da diferença** (`diferenca_pareada`): bootstrap pareado. Cada reamostragem sorteia um único
  conjunto de feedbacks, usado para A, para B e para todas as categorias; a diferença A − B é
  calculada dentro de cada reamostragem, o que permite também a diferença na média GP1–GP6. Só
  entram feedbacks com resposta válida em A e em B. Os arrays são `int8` para economizar memória.

### Divisão dos dados (`scripts/dividir_dados.py`)

- Cada feedback vai para o estrato da sua boa prática mais rara (ou "nenhuma"), e 20% de cada
  estrato vai para desenvolvimento. O número de cada estrato é arredondado pelo método dos maiores
  restos, para somar exatamente 200. Se os arquivos de `data/splits/` já existirem, nada é
  sobrescrito.

### Teste de consistência (`scripts/teste_consistencia.py`)

- O cliente do Claude só é importado no modo `--executar`, o único que chama a API (criar o cliente
  exige a `CLAUDE_API_KEY`). A coleta reaproveita `avaliar_modelo` de `gerar_respostas.py`, com o
  mesmo prompt e as mesmas tentativas, e também retoma de onde parou.

### App (`app.py`)

- Acurácia, precisão, recall e F1 (de 0 a 1) aparecem no gráfico como nota de 0 a 10; MCC e Kappa
  (de −1 a 1, sendo 0 o acaso) aparecem com o valor bruto.
- Categoria sem positivos no gabarito não ganha barra nas métricas indefinidas: o rótulo do eixo
  mostra "(N/A)", em vez de uma barra 0.
- Categorias com menos de 50 positivos no gabarito recebem aviso de intervalo largo. No teste de
  consistência, categorias com menos de 5 positivos nas duas execuções ficam em cinza, porque a
  concordância ali só mostra que as duas marcaram 0.
- As seções de intervalos, diferenças e comparação de prompts leem os CSVs gerados pelos scripts
  (`metricas_ic95.csv`, `diferencas_ic95.csv`, `comparacao_prompts.csv`) e só aparecem se eles
  existirem.
