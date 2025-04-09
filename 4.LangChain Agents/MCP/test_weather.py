# test_mcp_tools.py
from mcp_server_weather import MCPServer
import datetime
import requests
import os
from dotenv import load_dotenv


def calculate_distance_with_google(origin: str, destination: str, unit: str = "km"):
    """
    Calcula a distância entre duas localidades usando a Google Maps Distance Matrix API.

    Args:
        origin: Cidade ou endereço de origem
        destination: Cidade ou endereço de destino
        unit: Unidade de medida (km ou miles)

    Returns:
        String com a informação da distância
    """
    load_dotenv()  # Carrega a chave da API do arquivo .env
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")

    if not api_key:
        return "Erro: Chave da API do Google Maps não encontrada."

    # Define a unidade de medida para a API
    units = "imperial" if unit.lower() == "miles" else "metric"

    # Monta a URL da API
    base_url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        "origins": origin,
        "destinations": destination,
        "units": units,
        "key": api_key
    }

    try:
        # Faz a requisição para a API
        response = requests.get(base_url, params=params)
        data = response.json()

        # Verifica se a requisição foi bem-sucedida
        if data["status"] == "OK":
            # Extrai a distância
            distance = data["rows"][0]["elements"][0]["distance"]["text"]
            duration = data["rows"][0]["elements"][0]["duration"]["text"]

            return f"A distância entre {origin} e {destination} é de {distance}. O tempo estimado de viagem é {duration}."
        else:
            return f"Não foi possível calcular a distância: {data['status']}"

    except Exception as e:
        return f"Erro ao calcular a distância: {str(e)}"


def get_current_weather(location: str, unit: str = "celsius"):
    """
    Simula obter o clima atual para uma localização.
    Na implementação real, você chamaria uma API de clima.
    """
    # Simulação - em um caso real, você chamaria uma API de clima
    weather_data = {
        "São Paulo": {"temperature": 25, "condition": "ensolarado"},
        "Rio de Janeiro": {"temperature": 30, "condition": "parcialmente nublado"},
        "Brasília": {"temperature": 22, "condition": "chuvoso"},
        "default": {"temperature": 20, "condition": "desconhecido"}
    }

    result = weather_data.get(location, weather_data["default"])

    # Converter para Fahrenheit se necessário
    if unit.lower() == "fahrenheit":
        result["temperature"] = (result["temperature"] * 9 / 5) + 32

    return f"Clima em {location}: {result['temperature']}°{'F' if unit.lower() == 'fahrenheit' else 'C'}, {result['condition']}"


def get_current_time():
    """Retorna a hora atual."""
    now = datetime.datetime.now()
    return f"A hora atual é {now.strftime('%H:%M:%S')} em {now.strftime('%d/%m/%Y')}"


def search_web(query: str):
    """
    Simula uma pesquisa na web.
    Na implementação real, você usaria uma API de busca.
    """
    return f"Resultados da pesquisa para '{query}': [Simulação de resultados de busca]"


def test_mcp_with_tools():
    # Inicializa o servidor MCP
    mcp = MCPServer()

    # Registra ferramentas
    mcp.register_tool(
        tool_name="get_weather",
        tool_function=get_current_weather,
        description="Obtém o clima atual para uma localização específica",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "A cidade para obter informações do clima"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "A unidade de temperatura"
                }
            },
            "required": ["location"]
        }
    )

    mcp.register_tool(
        tool_name="calculate_distance",
        tool_function=calculate_distance_with_google,
        description="Calcula a distância e tempo de viagem entre duas localidades usando Google Maps",
        parameters={
            "type": "object",
            "properties": {
                "origin": {
                    "type": "string",
                    "description": "A cidade ou endereço de origem"
                },
                "destination": {
                    "type": "string",
                    "description": "A cidade ou endereço de destino"
                },
                "unit": {
                    "type": "string",
                    "enum": ["km", "miles"],
                    "description": "A unidade de distância (quilômetros ou milhas)"
                }
            },
            "required": ["origin", "destination"]
        }
    )

    mcp.register_tool(
        tool_name="get_time",
        tool_function=get_current_time,
        description="Obtém a hora atual",
        parameters={"type": "object", "properties": {}}
    )

    mcp.register_tool(
        tool_name="search_web",
        tool_function=search_web,
        description="Pesquisa informações na web",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A consulta de pesquisa"
                }
            },
            "required": ["query"]
        }
    )

    # Cria uma conversa com ferramentas
    conversation_id = "tools_test"
    mcp.create_conversation(conversation_id)

    # Adiciona as ferramentas à conversa
    mcp.add_tools_to_conversation(
        conversation_id, ["get_weather", "get_time", "search_web","calculate_distance"]
    )

    # Adiciona uma mensagem do sistema
    mcp.add_message(
        conversation_id=conversation_id,
        role="system",
        content="Você é um assistente útil que pode usar ferramentas para obter informações. Use as ferramentas quando apropriado."
    )

    # Adiciona uma mensagem do usuário
    mcp.add_message(
        conversation_id=conversation_id,
        role="user",
        content="Qual é a melhor rota de São Paulo para Rio de Janeiro? Quais estradas devo usar e quanto tempo vai levar?"
    )

    # Gera uma resposta (o modelo pode decidir usar ferramentas)
    response = mcp.generate_response(conversation_id)

    # Exibe a resposta
    print("Resposta final do assistente:")
    print(response["message"]["content"])

    # Exibe o histórico completo
    print("\nHistórico da conversa:")
    for msg in mcp.get_conversation_history(conversation_id):
        print(f"{msg['role']}: {msg.get('content', '')}")
        if 'tool_calls' in msg:
            print(
                f"  [Chamou ferramentas: {[tc.function.name for tc in msg['tool_calls']]}]"
            )


if __name__ == "__main__":
    test_mcp_with_tools()
