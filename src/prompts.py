import json
import re

CATEGORIAS = ["GP1", "GP2", "GP3", "GP4", "GP5", "GP6", "GP7"]

DESCRICOES_CATEGORIAS = {
    "GP1": {
        "nome": "Ajuda a esclarecer o bom desempenho",
        "definicao": (
            "O feedback dá exemplos eficazes de bom desempenho, mostra ao aluno quais são "
            "as metas, critérios e objetivos esperados, ou indica se o aluno atingiu (ou não) "
            "o objetivo esperado."
        ),
        "exemplo_positivo": (
            "Bom trabalho, mas vocês deveriam ter explorado os princípios de Alvaraz y Soler "
            "e os relacionados com as observações realizadas."
        ),
        "exemplo_negativo": "Bom trabalho, parabéns.",
    },
    "GP2": {
        "nome": "Facilita o desenvolvimento da auto-avaliação (reflexão) na aprendizagem",
        "definicao": (
            "O feedback solicita que o aluno compare sua atividade com a de outros colegas, "
            "peça a opinião do aluno sobre o próprio feedback recebido, ou peça que o aluno "
            "identifique os pontos fortes e fracos do próprio trabalho."
        ),
        "exemplo_positivo": (
            "Você fez um bom trabalho. Contudo, você pode melhorar a descrição sobre o "
            "assunto A. Que tal comparar a sua resposta com as dos outros colegas para saber "
            "como melhorar?"
        ),
        "exemplo_negativo": "Você errou a questão 3, corrija na próxima entrega.",
    },
    "GP3": {
        "nome": "Fornece informações de alta qualidade aos alunos sobre sua aprendizagem",
        "definicao": (
            "O feedback combina elogios com críticas construtivas, oferece conselhos "
            "corretivos específicos, e é objetivo/limitado em quantidade (poucos pontos, "
            "mas úteis) para que seja realmente utilizável pelo aluno."
        ),
        "exemplo_positivo": (
            "Bom trabalho, só atente-se mais na formatação de referências, como é mostrado "
            "no fascículo."
        ),
        "exemplo_negativo": "Está errado.",
    },
    "GP4": {
        "nome": "Incentiva o diálogo entre professores e colegas em torno da aprendizagem",
        "definicao": (
            "O feedback solicita que o aluno dialogue com os colegas de turma, ou ajuda o "
            "aluno a tirar dúvidas, abrindo espaço para conversa/interação sobre o tema."
        ),
        "exemplo_positivo": (
            "O seu trabalho não seguiu o que estava na descrição. Você está com dificuldades "
            "na disciplina? Pode me procurar para tirar dúvidas."
        ),
        "exemplo_negativo": "O trabalho não seguiu a descrição da atividade.",
    },
    "GP5": {
        "nome": "Incentiva crenças motivacionais positivas e auto-estima",
        "definicao": (
            "O feedback traz informações sobre progresso e realização, elogios ao aluno, "
            "e evita associar o retorno diretamente a uma nota."
        ),
        "exemplo_positivo": (
            "Parabéns a todos! O trabalho está irrepreensível! Quando o aluno da EaD tem "
            "compromisso com os estudos, com as atividades propostas, o resultado positivo "
            "sempre aparece."
        ),
        "exemplo_negativo": "Nota: 6,0. Revise o conteúdo.",
    },
    "GP6": {
        "nome": "Oferece oportunidades para fechar a lacuna entre o desempenho atual e o desejado",
        "definicao": (
            "O feedback ajuda o aluno a reconhecer os próximos passos no aprendizado e como "
            "realizá-los, indicando o que fazer para chegar ao nível desejado."
        ),
        "exemplo_positivo": (
            "Bom texto e boa leitura de imagens, faltou um pouco mais de cuidado com as "
            "questões de metodologia científica, espaçamento de margens, padrão das fontes e "
            "organização do trabalho. Na próxima vez é só consultar o fascículo de da "
            "disciplina de metodologia."
        ),
        "exemplo_negativo": "O trabalho tem problemas de formatação.",
    },
    "GP7": {
        "nome": "Fornece informações aos professores que podem ser usadas para ajudar a moldar o ensino",
        "definicao": (
            "O feedback solicita retorno do aluno sobre as atividades, ou verifica o nível "
            "de dificuldade de provas/atividades, gerando informação útil para o professor "
            "ajustar o ensino."
        ),
        "exemplo_positivo": (
            "Aluno X, você não respondeu a todas as questões. Você achou essas questões "
            "muito difíceis?"
        ),
        "exemplo_negativo": "Você não respondeu a questão 5.",
    },
}


IDS_EXEMPLOS_REVISADO = {
    "GP1": {"positivo": 13279, "negativo": 1119},
    "GP6": {"positivo": 1473, "negativo": 45},
}

DESCRICOES_CATEGORIAS_REVISADO = {
    **DESCRICOES_CATEGORIAS,
    "GP1": {
        "nome": DESCRICOES_CATEGORIAS["GP1"]["nome"],
        "definicao": (
            "O feedback avalia o trabalho em relação à proposta, aos critérios ou às exigências da "
            "atividade, dizendo se o aluno cumpriu (ou não) o que foi solicitado ou se atingiu o "
            "objetivo esperado, ou explica/exemplifica como deveria ser um bom desempenho. Apenas "
            "sugerir algo que poderia ter sido explorado, ou apontar o que faltou, sem se referir à "
            "proposta ou aos critérios da atividade, NÃO conta como GP1."
        ),
        "exemplo_positivo": "O trabalho de vocês cumpriu com o solicitado, porém senti falta da conclusão.",
        "exemplo_negativo": (
            "Boa tarde, Vocês poderiam ter explorado mais os princípios de Alvaraz y Sole e os "
            "relacionados pedagogicamente com as práticas observadas em sala de aula."
        ),
    },
    "GP6": {
        "nome": DESCRICOES_CATEGORIAS["GP6"]["nome"],
        "definicao": (
            "O feedback aponta erros, falhas ou o que faltou no trabalho (no conteúdo, na estrutura ou "
            "na formatação), ou indica o que o aluno deve fazer ou melhorar, dando a ele a informação "
            "necessária para reduzir a distância entre o desempenho atual e o desejado. Conta mesmo "
            "quando o feedback só aponta o que faltou, sem dizer explicitamente o próximo passo. "
            "Elogios sem nenhuma indicação de melhoria e avisos apenas administrativos (por exemplo, "
            "nome da equipe ou anexo não enviado) NÃO contam."
        ),
        "exemplo_positivo": (
            "No seu trabalho faltou o sumário, introdução, contextualização da escola, conclusão, "
            "referências e as fichas de comprovação de estágio."
        ),
        "exemplo_negativo": "Muito bom seu texto, foi construído com uma linguagem direta e coerente.",
    },
}

VERSOES_PROMPT = {
    "original": DESCRICOES_CATEGORIAS,
    "revisado": DESCRICOES_CATEGORIAS_REVISADO,
}


def montar_prompt(texto: str, versao: str = "original") -> str:
    if versao not in VERSOES_PROMPT:
        raise ValueError(f"versão de prompt desconhecida: {versao!r} (use {list(VERSOES_PROMPT)})")
    blocos = []
    for cat, info in VERSOES_PROMPT[versao].items():
        blocos.append(
            f"{cat} - {info['nome']}\n"
            f"Definição: {info['definicao']}\n"
            f"Exemplo que CONTA como {cat}: \"{info['exemplo_positivo']}\"\n"
            f"Exemplo que NÃO conta como {cat}: \"{info['exemplo_negativo']}\""
        )
    criterios = "\n\n".join(blocos)

    return f"""Você é um especialista em avaliação educacional formativa, aplicando o modelo dos 7 princípios de boas práticas de feedback (baseado em Nicol & Macfarlane-Dick).

Sua tarefa: analisar o feedback abaixo e decidir, para CADA categoria, se ela está PRESENTE (1) ou AUSENTE (0) no texto.

Critério de decisão: marque 1 apenas se houver evidência textual clara e específica daquela categoria, comparando com os exemplos fornecidos. Não infira intenção que não está escrita no texto. Um feedback pode conter zero, uma ou várias categorias ao mesmo tempo — elas não são mutuamente exclusivas.

CATEGORIAS:

{criterios}

FEEDBACK A ANALISAR:
\"\"\"{texto}\"\"\"

Responda APENAS em JSON válido, sem texto adicional antes ou depois, no formato:
{{"GP1": 0, "GP2": 0, "GP3": 0, "GP4": 0, "GP5": 0, "GP6": 0, "GP7": 0}}
"""


def parse_resposta(resposta: str) -> dict[str, int | None]:
    texto = resposta.strip()
    texto = re.sub(r"^```(?:json)?\s*|\s*```$", "", texto, flags=re.MULTILINE).strip()

    try:
        dados = json.loads(texto)
        return {
            categoria: (
                int(dados[categoria])
                if categoria in dados and str(dados[categoria]) in ("0", "1")
                else None
            )
            for categoria in CATEGORIAS
        }
    except (json.JSONDecodeError, TypeError):
        pass

    valores: dict[str, int | None] = {}
    for categoria in CATEGORIAS:
        match = re.search(rf'"?{categoria}"?\s*:\s*([01])', resposta)
        valores[categoria] = int(match.group(1)) if match else None
    return valores


if __name__ == "__main__":
    exemplo = (
        "Bom trabalho, mas vocês deveriam ter explorado os princípios de Alvaraz y Soler "
        "e os relacionados com as observações realizadas. Que tal comparar sua resposta "
        "com a de outros colegas?"
    )
    print(montar_prompt(exemplo))
