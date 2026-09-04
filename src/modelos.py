"""Registro central dos modelos comparados no projeto.

Cada chave é o id interno do modelo — usado no nome do módulo cliente
(`src/<id>_client.py`) e no nome do arquivo de resultados
(`data/processed/respostas_<id>.csv`).
"""

MODELOS = {
    "claude": {"nome": "Claude", "cor": "#2a78d6"},
    "gemini": {"nome": "Gemini", "cor": "#eb6834"},
    "openai": {"nome": "OpenAI", "cor": "#1baf7a"},
}
