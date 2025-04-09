# mcp_server.py
import os
import logging
from typing import Dict, List, Optional, Union
from openai import OpenAI
from dotenv import load_dotenv


class MCPServer:
    def __init__(self, api_key: Optional[str] = None):
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('mcp_server')

        # Carregar variáveis de ambiente
        load_dotenv()

        # Configurar API key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Provide it as an argument or set OPENAI_API_KEY environment variable.")

        # Inicializar cliente OpenAI
        self.client = OpenAI(api_key=self.api_key)

        # Inicializar estruturas de dados
        self.conversations = {}
        self.conversation_models = {}

        # Definir modelos padrão
        self.models = {
            "default": "gpt-4-turbo-preview",
            "fast": "gpt-3.5-turbo",
            "vision": "gpt-4-vision-preview"
        }

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

        self.logger.info(f"Created new conversation: {conversation_id}")
        return conversation_id

    def add_message(self, conversation_id: str, role: str, content: Union[str, List], name: Optional[str] = None) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            conversation_id: ID of the conversation
            role: Role of the message sender (system, user, assistant)
            content: Content of the message
            name: Optional name for the message sender
        """
        # Create conversation if it doesn't exist
        if conversation_id not in self.conversations:
            self.create_conversation(conversation_id)

        # Create message object
        message = {"role": role, "content": content}
        if name:
            message["name"] = name

        # Add to conversation history
        self.conversations[conversation_id].append(message)

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

    def set_model(self, alias: str, model_name: str) -> None:
        """
        Set or update a model alias.
        
        Args:
            alias: Alias for the model (e.g., "default", "fast")
            model_name: Full name of the model (e.g., "gpt-4-turbo-preview")
        """
        self.models[alias] = model_name

    def generate_response(self, conversation_id: str, model: Optional[str] = None,
                          temperature: float = 0.7, max_tokens: Optional[int] = None) -> Dict:
        """
        Generate a response from the model based on conversation history.
        
        Args:
            conversation_id: ID of the conversation
            model: Model to use (overrides the one set for the conversation)
            temperature: Temperature parameter for generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary containing the response and metadata
        """
        # Create conversation if it doesn't exist
        if conversation_id not in self.conversations:
            self.create_conversation(conversation_id)

        # Get the model to use
        model_to_use = model or self.conversation_models.get(
            conversation_id, self.models["default"])

        # Prepare the messages
        messages = self.get_conversation_history(conversation_id)

        # Prepare parameters
        params = {
            "model": model_to_use,
            "messages": messages,
            "temperature": temperature
        }

        if max_tokens:
            params["max_tokens"] = max_tokens

        self.logger.info(
            f"Generating response using {model_to_use} for conversation {conversation_id}")

        try:
            # Call the OpenAI API with the new client format (synchronous)
            response = self.client.chat.completions.create(**params)

            # Extract the response message
            message_content = response.choices[0].message.content

            # Add the response to the conversation history
            self.add_message(conversation_id, "assistant", message_content)

            # Return the response
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
