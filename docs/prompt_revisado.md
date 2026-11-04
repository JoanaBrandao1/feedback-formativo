# Experimento complementar: prompt revisado (exploratório)

O **experimento principal** do TCC usa o prompt original (`montar_prompt(texto)`, versão
`"original"`) nos 1000 feedbacks. Este documento descreve um **experimento complementar e
exploratório**, que testa se alinhar as definições de GP1 e GP6 à forma como os anotadores
aplicaram o guia melhora a concordância dos modelos.

## Por que revisar só GP1 e GP6

As definições e os exemplos positivos do prompt original vieram do guia de anotação
(veja `docs/execucao.md`, seção "Origem do prompt"). Os modelos seguem esse texto ao pé da
letra, mas nessas duas categorias os anotadores aplicaram o guia de outro jeito:

| | O que o guia diz | Como os anotadores aplicaram | Efeito nos modelos (rodada completa) |
|---|---|---|---|
| **GP1** | "mostrar metas, critérios e objetivos esperados… se o aluno atingiu o objetivo"; exemplo: feedback "Alvaraz y Soler" | De forma **mais restrita**: os 22 feedbacks parecidos com o exemplo do guia têm GP1 = 0 no gabarito | Marcam GP1 demais (Claude: 561 contra 125 no gabarito) |
| **GP6** | "reconhecer os próximos passos no aprendizado e como realizá-los" | De forma **mais ampla**: também quando o feedback só aponta erros ou o que faltou (CAVALCANTI et al., 2023: "indicar os erros para o aluno evitar repeti-los") | Marcam GP6 de menos (Claude: 167 contra 603 no gabarito) |

Além disso, o exemplo **negativo** de GP6 do prompt original ("O trabalho tem problemas de
formatação."), que não vem do guia, ensina o oposto da prática dos anotadores.

## Como evitar ajustar o prompt no próprio gabarito

1. `scripts/dividir_dados.py` dividiu os 1000 feedbacks **uma única vez** (semente 42,
   estratificada pela boa prática mais rara de cada feedback):
   - **desenvolvimento (200 IDs)**, em `data/splits/dev_ids.csv`: os únicos feedbacks lidos
     para escrever o prompt revisado;
   - **teste (800 IDs)**, em `data/splits/teste_ids.csv`: onde os dois prompts são comparados.
     Nenhum texto ou rótulo do teste foi usado para escrever a revisão.
2. O prompt original **não é rodado de novo**: as respostas dele nos 800 IDs de teste já
   existem nos `respostas_<modelo>.csv`.
3. A comparação usa bootstrap **pareado** (os mesmos feedbacks sorteados para as duas versões).

Positivos em cada parte:

| | GP1 | GP2 | GP3 | GP4 | GP5 | GP6 | GP7 |
|---|---|---|---|---|---|---|---|
| Desenvolvimento (200) | 25 | 5 | 59 | 2 | 45 | 125 | 0 |
| Teste (800) | 100 | 24 | 249 | 11 | 189 | 478 | 0 |

## O que mudou no prompt

Só a definição e os dois exemplos de **GP1** e **GP6** (`DESCRICOES_CATEGORIAS_REVISADO` em
`src/prompts.py`). O cabeçalho, as instruções, as outras cinco categorias e o formato JSON são
idênticos ao original.

- **GP1:** passa a exigir que o feedback **avalie o trabalho em relação à proposta, aos
  critérios ou às exigências da atividade** (se cumpriu ou não o solicitado, se atingiu o
  objetivo) ou que explique/exemplifique um bom desempenho. Apenas sugerir algo que poderia ter
  sido explorado, ou apontar o que faltou, sem se referir à proposta, **não conta**.
- **GP6:** passa a contar quando o feedback **aponta erros, falhas ou o que faltou**, ou indica
  o que melhorar, **mesmo sem dizer explicitamente o próximo passo**. Elogios sem indicação de
  melhoria e avisos apenas administrativos (nome da equipe, anexo não enviado) não contam.

Exemplos novos (todos do conjunto de desenvolvimento, com o rótulo do gabarito):

| Categoria | Tipo | ID | Texto | Gabarito |
|---|---|---|---|---|
| GP1 | positivo | 13279 | "O trabalho de vocês cumpriu com o solicitado, porém senti falta da conclusão." | GP1 = 1 |
| GP1 | negativo | 1119 | "Boa tarde, Vocês poderiam ter explorado mais os princípios de Alvaraz y Sole e os relacionados pedagogicamente com as práticas observadas em sala de aula." | GP1 = 0 |
| GP6 | positivo | 1473 | "No seu trabalho faltou o sumário, introdução, contextualização da escola, conclusão, referências e as fichas de comprovação de estágio." | GP6 = 1 |
| GP6 | negativo | 45 | "Muito bom seu texto, foi construído com uma linguagem direta e coerente." | GP6 = 0 |

**PREENCHER:** confirmação do orientador sobre como os anotadores aplicaram GP1 e GP6 (se ele
concordar com a leitura acima, ou se indicar outro critério, a revisão deve seguir a orientação
dele e ser feita antes de qualquer chamada à API).

## Como rodar

```bash
python scripts/dividir_dados.py                    # já foi feito; não sobrescreve
python scripts/gerar_respostas.py --prompt revisado --ids data/splits/teste_ids.csv --plano    # custo, sem API
python scripts/gerar_respostas.py --prompt revisado --ids data/splits/teste_ids.csv --modelo gemini
python scripts/gerar_respostas.py --prompt revisado --ids data/splits/teste_ids.csv --modelo openai
python scripts/gerar_respostas.py --prompt revisado --ids data/splits/teste_ids.csv --modelo claude
python scripts/comparar_prompts.py                 # sem API → data/processed/comparacao_prompts.csv
```

Estimativa do `--plano` (800 chamadas por modelo, preços de referência a conferir): Claude
~US$ 2,90, Gemini ~US$ 0,15, OpenAI ~US$ 0,20; cerca de 50 minutos por modelo.

## Como relatar no TCC

- Como **experimento exploratório**, separado do resultado principal.
- A revisão foi orientada pelos anotadores/orientador e pelo conjunto de desenvolvimento; a
  avaliação foi feita só no conjunto de teste.
- O Claude não é determinístico (sem `temperature`), então parte de qualquer diferença do Claude
  pode ser variação entre execuções (veja o teste de consistência em `docs/execucao.md`).
