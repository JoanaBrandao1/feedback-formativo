# Metadados de execução

Registro de **como** as respostas dos modelos em `data/processed/respostas_<modelo>.csv`
foram geradas. Foi montado a partir do código (`src/*_client.py`, `src/prompts.py`,
`scripts/gerar_respostas.py`), do `requirements.txt` e do histórico do git. Onde
nada disso comprova a informação, o campo está marcado como **PREENCHER**.

## Configuração por modelo

| | Claude | Gemini | OpenAI |
|---|---|---|---|
| Identificador na API | `claude-sonnet-5` | `gemini-3.1-flash-lite` | `gpt-4o-mini` |
| Biblioteca (versão do `requirements.txt`) | `anthropic` 0.122.0 | `google-genai` 2.4.0 | `openai` 2.37.0 |
| Temperatura | **não enviada** (veja nota 1) | `0` | `0` |
| Saída forçada em JSON? | **Não** (só pedido no prompt) | Sim (`response_mime_type="application/json"`) | Sim (`response_format={"type": "json_object"}`) |
| Limite de tokens de saída | `max_tokens=1024` | padrão da API | padrão da API |
| Respostas com cerca ` ```json ` | 242 de 1000 (o parser remove) | 0 | 0 |
| Falhas de parsing / respostas incompletas (CSV atual) | 0 | 0 | 0 |

**Nota 1 (temperatura do Claude).** A API rejeita o parâmetro `temperature` para o
`claude-sonnet-5` (erro 400, "deprecated for this model"), então ele **não é enviado**
e o modelo usa a amostragem padrão. Por isso o Claude **não é determinístico**: rodar de
novo pode dar respostas diferentes. Gemini e OpenAI usam `temperature=0`, que reduz mas
não garante respostas idênticas entre execuções.

**Nota 3 (modelos de níveis diferentes).** O `claude-sonnet-5` é um modelo de nível superior
aos outros dois (`gemini-3.1-flash-lite` e `gpt-4o-mini` são versões menores e mais baratas
das respectivas empresas). Além disso, o Claude roda sem temperatura 0 e sem JSON forçado. A
comparação é, portanto, entre **essas configurações**, e não entre "o melhor modelo de cada
empresa". Isso deve constar nas limitações do TCC.

## Comum aos três modelos

- **Prompt:** `montar_prompt()` em `src/prompts.py` — definição de cada GP, um exemplo
  positivo e um negativo por categoria, e pedido de resposta só em JSON (origem detalhada na
  seção "Origem do prompt", abaixo). É o mesmo texto
  para os três modelos. O histórico do git tem uma única versão desse arquivo
  (commit `e6fd394`, 2026-09-04); não dá para provar pelo git se havia alterações não
  commitadas quando cada modelo rodou. Os arquivos `*_prompt_antigo.csv` são de um teste
  com prompt anterior (10 linhas) e não entram na análise. Em novembro de 2026,
  `src/prompts.py` ganhou uma segunda versão (`"revisado"`, para o experimento complementar);
  a versão `"original"`, que é o padrão, continua gerando exatamente o mesmo texto (conferido
  por hash nos 1000 feedbacks).
- **Uma chamada por feedback**, independente das outras (sem histórico de conversa),
  em sequência, com pausa de 1 s entre chamadas.
- **Tentativas:** até 3 por feedback (5 s de espera entre elas) em caso de erro de API.
  Resposta mal formatada **não** gera nova tentativa.
- **Dataset:** `data/raw/dataset.csv`, 1000 feedbacks com o gabarito humano, anotado por
  especialistas: dois anotadores (concordância κ = 0,43 nas boas práticas, antes do consenso)
  e os casos de divergência resolvidos por outros dois especialistas (CAVALCANTI et al., 2023).
  O arquivo também traz as colunas FT, FP, FR e FS (níveis de Hattie & Timperley), que **não
  são usadas**: por decisão do orientador, o trabalho avalia só as boas práticas (GP1–GP7).

## Origem do prompt

- **Definições e exemplos positivos:** copiados do guia de anotação "Pontos de avaliação do
  feedback" (documento Word, 2018), usado na anotação do dataset. **PREENCHER:** autor e forma
  de citação do guia (confirmar com o orientador).
- **Exemplos negativos** ("Exemplo que NÃO conta como…"): **não estão no guia**; foram
  elaborados pela autora. O de GP6 ("O trabalho tem problemas de formatação.") vai contra a
  forma como os anotadores aplicaram GP6 (veja `docs/prompt_revisado.md`).
- **Exemplos que estão no próprio dataset avaliado:** o exemplo positivo de GP3 é o feedback
  ID 9361 (no gabarito: GP3 = 0, GP5 = 1, GP6 = 1) e o exemplo positivo de GP6 é o ID 13628
  (no gabarito: GP1, GP3, GP5 e GP6 = 1). Os demais exemplos não aparecem no dataset. O impacto
  de manter esses 2 feedbacks na avaliação é desprezível:

| Modelo | Métrica (média GP1–GP6) | Com os 1000 | Sem os IDs 9361 e 13628 | Diferença |
|---|---|---|---|---|
| Claude | F1 / MCC / Kappa | 0,437 / 0,354 / 0,321 | 0,436 / 0,353 / 0,320 | −0,001 / −0,001 / −0,001 |
| Gemini | F1 / MCC / Kappa | 0,485 / 0,363 / 0,323 | 0,485 / 0,363 / 0,322 | −0,001 / 0,000 / 0,000 |
| OpenAI | F1 / MCC / Kappa | 0,470 / 0,384 / 0,358 | 0,470 / 0,385 / 0,359 | 0,000 / +0,001 / +0,001 |

  Por isso, a análise principal mantém os 1000 feedbacks.

## Datas de geração

O CSV não guarda data/hora. As datas abaixo foram **inferidas** do horário de modificação
dos arquivos e do histórico do git, e registradas aqui com a autorização da autora. São
datas de conclusão da coleta (fuso -03:00), não de início.

| Modelo | Data de geração | Base da informação |
|---|---|---|
| Claude (1ª sessão, 859 linhas) | até **2026-09-04** | Essas linhas já estão no commit `e6fd394` (2026-09-04 09:27); o horário exato de coleta não é conhecido |
| Claude (2ª sessão, 141 linhas refeitas) | **2026-09-21** | Arquivo modificado em 2026-09-21 14:19; commit `ba2b5d5` |
| Gemini | **2026-09-04** | Arquivo modificado em 2026-09-04 05:05 |
| OpenAI | **2026-09-04** | Arquivo modificado em 2026-09-04 05:39 |

**Nota 2 (Claude gerado em duas sessões).** A versão de `respostas_claude.csv` no commit
`e6fd394` (2026-09-04) tinha **141 linhas incompletas**: 140 com `resposta_bruta` vazia
(falha de API) e 1 truncada (ID 9011). Essas 141 linhas foram refeitas depois (arquivo
modificado em 2026-09-21), e as outras 859 permaneceram idênticas. O arquivo atual está
completo (0 falhas), mas é uma mistura de duas sessões de coleta, com a mesma configuração
descrita acima. Como o Claude não é determinístico e o identificador do modelo não tem data,
a autora acredita que o modelo servido foi o mesmo nas duas sessões; isso **não é
verificável** (a versão exata servida não foi salva) e fica registrado como suposição.
Gemini e OpenAI não tiveram esse problema (CSVs idênticos entre o commit de 04/09 e hoje).

## O que não foi registrado (e não dá para recuperar dos CSVs)

- **Versão exata do modelo servido.** O código usa o identificador sem sufixo de data
  (por exemplo, `gpt-4o-mini` é um apelido que a OpenAI aponta para uma versão datada).
  A API devolve o nome exato usado em cada resposta, mas esse campo não foi salvo.
  Se quiser registrar no TCC, indique o identificador acima e a data de coleta.
- **Versões das bibliotecas na hora da coleta.** O `requirements.txt` mostra as versões
  instaladas hoje, que podem não ser as mesmas da coleta.
- **Uso de tokens e custo** de cada rodada.

## Análise estatística

| | |
|---|---|
| Métricas | Acurácia, precisão, recall, F1, MCC e Kappa de Cohen (scikit-learn 1.8.0), por categoria |
| Resumo geral | Média simples das categorias **com positivos no gabarito** (GP1–GP6); GP7 não tem nenhum positivo e fica como N/A |
| Intervalo de confiança | Bootstrap percentil de 95%, **1000 reamostragens**, semente **42** (por categoria: `default_rng([42, índice da categoria])`), para F1, MCC e Kappa — `scripts/calcular_ic95.py` → `data/processed/metricas_ic95.csv` |
| Diferença entre modelos | IC 95% da diferença A − B para cada par de modelos, por **bootstrap pareado**: 1000 reamostragens, semente 42, cada uma sorteando os mesmos feedbacks para os dois modelos e para todas as categorias; a diferença é calculada dentro de cada reamostragem, por categoria e na média GP1–GP6. Só entram os feedbacks que os dois modelos responderam. Diferença significativa = IC não contém 0 (nível de 5%), sem correção para comparações múltiplas (54 intervalos por categoria: 3 pares × 6 categorias × 3 métricas, mais 9 da média) — `scripts/calcular_ic95.py` → `data/processed/diferencas_ic95.csv` |
| Casos indefinidos | Precisão/recall/F1 indefinidos valem 0 (`zero_division=0`); MCC e Kappa indefinidos valem 0 |
| Python | 3.14.5 |

## Teste de consistência do Claude

Como o Claude roda sem `temperature=0`, as mesmas 100 perguntas foram feitas de novo para
medir o quanto ele concorda com ele mesmo. **Executado em 2026-10-16** com
`python scripts/teste_consistencia.py --executar`: 100 feedbacks sorteados (semente 42),
mesmo prompt e mesma configuração, salvos em `data/processed/consistencia_claude.csv`
(resumo em `consistencia_claude_resumo.csv`). Todas as 100 respostas foram lidas por completo;
houve 1 erro passageiro de API (ID 11869) que passou na segunda tentativa.

| Categoria | Concordância entre as 2 execuções | Kappa | Positivos (original → nova) |
|---|---|---|---|
| GP1 | 90% | 0,80 | 56 → 58 |
| GP2 | 100% | N/A | 0 → 0 |
| GP3 | 92% | 0,81 | 30 → 28 |
| GP4 | 100% | 1,00 | 1 → 1 |
| GP5 | 94% | 0,76 | 16 → 14 |
| GP6 | 93% | 0,77 | 17 → 20 |
| GP7 | 100% | N/A | 0 → 0 |
| **Os 7 GP idênticos no mesmo feedback** | **72%** | | 28 dos 100 feedbacks tiveram pelo menos um GP diferente |

**Como ler.** Nas categorias com positivos suficientes (GP1, GP3, GP5, GP6), o Claude repete a
própria decisão em 90–94% dos casos (Kappa 0,76–0,81). As mudanças vão nos dois sentidos
(0→1 e 1→0) e os totais de positivos quase não se alteram, então não há deriva sistemática.

**Limites.**
- GP2, GP4 e GP7 **não são avaliadas** por este teste: a amostra tem 0, 1 e 0 positivos do Claude.
  O 100% nessas categorias só mostra que as duas execuções marcaram 0. O Kappa 1,00 do GP4 vem de um único caso.
- É uma única repetição (2 execuções) em 100 feedbacks.
- Só o Claude foi testado. Gemini e OpenAI usam `temperature=0`, que reduz mas não garante respostas idênticas.
- O bootstrap (IC 95%) só captura a variação por trocar os feedbacks; a variação por rodar o modelo
  de novo é uma incerteza a mais, que não está nos intervalos.
