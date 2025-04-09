# mcp_server_weather.py
import os
import logging
import json
from typing import Dict, List, Optional, Union, Callable, Any
from openai import OpenAI
from dotenv import load_dotenv
import requests


class MCPServer:
    def __init__(self, api_key: Optional[str] = None):
        # Configuração inicial
        load_dotenv()
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('mcp_server')

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Provide it as an argument or set OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=self.api_key)

        # Estruturas de dados existentes
        self.conversations = {}
        self.conversation_models = {}
        self.models = {
            "default": "gpt-4-turbo-preview",
            "fast": "gpt-3.5-turbo",
            "vision": "gpt-4-vision-preview"
        }

        # Nova estrutura para gerenciar ferramentas
        self.tools = {}
        self.conversation_tools = {}

        self.logger.info("MCP Server initialized successfully")

    def create_conversation(self, conversation_id: str, model: Optional[str] = None) -> str:
        """
        Create a new conversation with the specified ID.

        Args:
            conversation_id: Unique identifier for the conversation
            model: Model alias to use for this conversation (default, fast, vision, etc.)

        Returns:
            The conversation ID
        """
        model_name = model or self.models["default"]

        # Initialize the conversation history
        self.conversations[conversation_id] = []

        # Store the model being used for this conversation
        self.conversation_models[conversation_id] = model_name

        return conversation_id

    def get_conversation_history(self, conversation_id: str) -> List[Dict]:
        """
        Get the full history of a conversation.

        Args:
            conversation_id: ID of the conversation

        Returns:
            List of message dictionaries
        """
        if conversation_id not in self.conversations:
            return []
        return self.conversations[conversation_id]

    def clear_conversation(self, conversation_id: str) -> None:
        """
        Clear the history of a conversation.

        Args:
            conversation_id: ID of the conversation to clear
        """
        if conversation_id in self.conversations:
            self.conversations[conversation_id] = []

    def register_tool(self, tool_name: str, tool_function: Callable,
                     description: str, parameters: Dict) -> None:
        """
        Registra uma nova ferramenta que pode ser usada pelo modelo.

        Args:
            tool_name: Nome único da ferramenta
            tool_function: Função Python que implementa a ferramenta
            description: Descrição do que a ferramenta faz
            parameters: Esquema JSON dos parâmetros da ferramenta
        """
        tool_definition = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": description,
                "parameters": parameters
            }
        }

        self.tools[tool_name] = {
            "definition": tool_definition,
            "function": tool_function
        }

        self.logger.info(f"Registered tool: {tool_name}")

    def add_tools_to_conversation(self, conversation_id: str, tool_names: List[str]) -> None:
        """
        Adiciona ferramentas específicas a uma conversa.

        Args:
            conversation_id: ID da conversa
            tool_names: Lista de nomes de ferramentas a serem adicionadas
        """
        if conversation_id not in self.conversations:
            self.create_conversation(conversation_id)

        # Verifica se todas as ferramentas existem
        for tool_name in tool_names:
            if tool_name not in self.tools:
                raise ValueError(f"Tool '{tool_name}' is not registered")

        # Adiciona as ferramentas à conversa
        self.conversation_tools[conversation_id] = tool_names
        self.logger.info(f"Added tools {tool_names} to conversation {conversation_id}")

    def _execute_tool(self, tool_name: str, arguments: str) -> Any:
        """
        Executa uma ferramenta com os argumentos fornecidos.

        Args:
            tool_name: Nome da ferramenta a ser executada
            arguments: Argumentos em formato JSON string

        Returns:
            Resultado da execução da ferramenta
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' is not registered")

        tool = self.tools[tool_name]
        function = tool["function"]

        # Converte os argumentos de string JSON para dicionário
        try:
            args_dict = json.loads(arguments)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON arguments: {arguments}")

        # Executa a função da ferramenta
        self.logger.info(f"Executing tool: {tool_name} with arguments: {arguments}")
        result = function(**args_dict)
        return result

    def generate_response(self, conversation_id: str, model: Optional[str] = None,
                         temperature: float = 0.7, max_tokens: Optional[int] = None) -> Dict:
        """
        Gera uma resposta do modelo com base no histórico da conversa,
        potencialmente usando ferramentas.

        Args:
            conversation_id: ID da conversa
            model: Modelo a ser usado (substitui o definido para a conversa)
            temperature: Parâmetro de temperatura para geração
            max_tokens: Tokens máximos a serem gerados

        Returns:
            Dicionário contendo a resposta e metadados
        """
        # Cria a conversa se não existir
        if conversation_id not in self.conversations:
            self.create_conversation(conversation_id)

        # Obtém o modelo a ser usado
        model_to_use = model or self.conversation_models.get(conversation_id, self.models["default"])

        # Prepara as mensagens
        messages = self.get_conversation_history(conversation_id)

        # Prepara os parâmetros
        params = {
            "model": model_to_use,
            "messages": messages,
            "temperature": temperature
        }

        if max_tokens:
            params["max_tokens"] = max_tokens

        # Adiciona ferramentas se estiverem configuradas para esta conversa
        if conversation_id in self.conversation_tools:
            tool_names = self.conversation_tools[conversation_id]
            tools = [self.tools[name]["definition"] for name in tool_names]
            params["tools"] = tools

        self.logger.info(f"Generating response using {model_to_use} for conversation {conversation_id}")

        try:
            # Chama a API OpenAI
            response = self.client.chat.completions.create(**params)

            # Processa a resposta
            message = response.choices[0].message

            # Verifica se o modelo quer usar uma ferramenta
            if hasattr(message, 'tool_calls') and message.tool_calls:
                self.logger.info(f"Model requested to use tools: {message.tool_calls}")

                # Adiciona a mensagem do assistente solicitando a ferramenta
                self.add_message(conversation_id, "assistant", message.content,
                                tool_calls=message.tool_calls)

                # Processa cada chamada de ferramenta
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = tool_call.function.arguments

                    # Executa a ferramenta
                    tool_result = self._execute_tool(function_name, function_args)

                    # Adiciona o resultado da ferramenta ao histórico
                    self.add_message(
                        conversation_id=conversation_id,
                        role="tool",
                        content=str(tool_result),
                        tool_call_id=tool_call.id
                    )

                # Gera uma nova resposta com os resultados das ferramentas
                return self.generate_response(conversation_id, model, temperature, max_tokens)
            else:
                # Resposta normal sem uso de ferramentas
                message_content = message.content

                # Adiciona a resposta ao histórico da conversa
                self.add_message(conversation_id, "assistant", message_content)

                # Retorna a resposta
                return {
                    "message": {"content": message_content, "role": "assistant"},
                    "model": model_to_use,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                }
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            raise

    def add_message(self, conversation_id: str, role: str, content: Union[str, List],
                   name: Optional[str] = None, tool_calls: Optional[List] = None,
                   tool_call_id: Optional[str] = None) -> None:
        """
        Adiciona uma mensagem ao histórico da conversa.

        Args:
            conversation_id: ID da conversa
            role: Papel do remetente da mensagem (system, user, assistant, tool)
            content: Conteúdo da mensagem
            name: Nome opcional para o remetente da mensagem
            tool_calls: Chamadas de ferramentas (para mensagens do assistente)
            tool_call_id: ID da chamada de ferramenta (para mensagens de ferramenta)
        """
        # Cria a conversa se não existir
        if conversation_id not in self.conversations:
            self.create_conversation(conversation_id)

        # Cria o objeto da mensagem
        message = {"role": role, "content": content}
        if name:
            message["name"] = name
        if tool_calls:
            message["tool_calls"] = tool_calls
        if tool_call_id:
            message["tool_call_id"] = tool_call_id

        # Adiciona ao histórico da conversa
        self.conversations[conversation_id].append(message)


# Função de utilidade fora da classe (mova para dentro da classe se necessário)
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