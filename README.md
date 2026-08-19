# feedback-formativo

Classificação de feedbacks educacionais com IA (testes exploratórios).

## Notebooks

| Arquivo | Provedor |
|---------|----------|
| `notebooks/analise_openai.ipynb` | OpenAI (`gpt-4o-mini`) |
| `notebooks/analise_gemini.ipynb` | Google Gemini (`gemini-2.5-flash`) |
| `notebooks/analise_claude.ipynb` | Anthropic Claude (`claude-sonnet-5`) |

Os três notebooks incluem, além da chamada simples, uma seção de **processamento em
lote** (Batch API): em vez de chamar a API uma vez por linha do dataset, todas as
requisições são enviadas de uma vez e processadas de forma assíncrona, a 50% do custo
normal. Por padrão, a seção de lote roda o dataset **completo (1000 feedbacks)** — o
processamento não é instantâneo (pode levar de minutos a até 24h, dependendo do
provedor), então essas células foram deixadas sem executar nos notebooks salvos.

## Como rodar

```bash
cd feedback-formativo
source .venv/Scripts/activate   # Git Bash
pip install -r requirements.txt
```

Configure `.env` (veja `.env.example`):

```
OPENAI_API_KEY=...
GEMINI_API_KEY=...
CLAUDE_API_KEY=...
```

Inicie o Jupyter:

```bash
.venv/Scripts/python.exe -m jupyter notebook
```

Abra o notebook do provedor que quiser testar e execute as células em ordem.
