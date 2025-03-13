# pip install -U langgraph

import subprocess
import re
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import uuid
from typing import Annotated, List
from typing_extensions import TypedDict

# Inicializar o LLM
llm = ChatOllama(model="qwen2.5:latest")

# Lista de comandos permitidos para maior segurança
ALLOWED_COMMANDS = ["ls", "pwd", "whoami", "date",
                    "cat", "echo", "grep", "find", "wc", "head", "tail", "hostname"]


def is_safe_command(command: str) -> bool:
    """Verifica se o comando é seguro para executar"""
    base_cmd = command.split()[0]
    return base_cmd in ALLOWED_COMMANDS


def run_shell_command(command: str) -> str:
    """Executa um comando shell com verificação de segurança"""
    if not is_safe_command(command):
        return f"Comando não permitido por razões de segurança: {command}"

    try:
        result = subprocess.run(command, shell=True,
                                check=True, text=True, capture_output=True)
        return result.stdout.strip() if result.stdout else "Comando executado com sucesso."
    except subprocess.CalledProcessError as e:
        return f"Erro ao executar o comando: {e}\nOutput de erro: {e.stderr}"


def convert_to_shell_command(natural_language: str) -> str:
    """Converte linguagem natural para um comando shell usando LLM"""
    prompt = f"""
    Converta o seguinte PEDIDO em linguagem natural para um comando válido em linux.
    Retorne apenas o comando shell sem explicações.
    EXEMPLO:
    - Input: "Listar todos os ficheiros"
    - Output: "ls"
    PEDIDO: {natural_language}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    command = response.content.strip()

    # Remove backticks if present (common LLM output format)
    command = re.sub(r'^```.*\n|```$', '', command).strip()

    return command


def process_shell_tool(state):
    natural_language = state["messages"][-1].content.strip()
    command = convert_to_shell_command(natural_language)
    output = run_shell_command(command)

    return {"messages": [AIMessage(content=f"Comando: {command}\n\nOutput:\n{output}")]}


class State(TypedDict):
    messages: Annotated[List[HumanMessage |
                             AIMessage], "Histórico de mensagens"]


workflow = StateGraph(State)
workflow.add_node("process_shell_tool", process_shell_tool)

workflow.add_edge(START, "process_shell_tool")
workflow.add_edge("process_shell_tool", END)

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": str(uuid.uuid4())}}

if __name__ == '__main__':
    print("Assistente de comandos shell (digite 'exit' para sair)")
    print("Comandos permitidos:", ", ".join(ALLOWED_COMMANDS))

    while True:
        user_input = input("\nUtilizador: ")
        if user_input.lower() in {'q', 'quit', 'exit', 'sair'}:
            print("Adeus!")
            break

        output = graph.invoke(
            {"messages": [HumanMessage(content=user_input)]}, config=config)
        print("IA:", output["messages"][-1].content)
