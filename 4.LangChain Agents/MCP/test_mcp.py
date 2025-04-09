# test_mcp.py
from mcp_server import MCPServer


def test_conversation():
    # Inicializa o servidor MCP
    mcp = MCPServer()

    # Cria uma nova conversa
    conversation_id = "test_conversation"
    mcp.create_conversation(conversation_id)

    # Adiciona uma mensagem do sistema
    mcp.add_message(
        conversation_id=conversation_id,
        role="system",
        content="Você é um especialista em gerar jogos para terapia da fala. Esses jogos são em formato que quizz"
    )

    # Adiciona uma mensagem do usuário
    mcp.add_message(
        conversation_id=conversation_id,
        role="user",
        content="Gere um trava lingua"
    )

    # Gera uma resposta (agora síncrona)
    response = mcp.generate_response(conversation_id)

    # Exibe a resposta
    print("Resposta do assistente:")
    print(response["message"]["content"])

    # Exibe o histórico completo
    print("\nHistórico da conversa:")
    for msg in mcp.get_conversation_history(conversation_id):
        print(f"{msg['role']}: {msg['content']}")


# Executa o teste
if __name__ == "__main__":
    test_conversation()
