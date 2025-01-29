import sqlite3
from collections import deque
from typing import List
# Imports para LLM via Hugging Face
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    set_seed
)

# ---------------------------------------------------------
# 1) CONFIGURAÇÕES DO MODELO LOCAL (Ex.: Falcon 7B Instruct)
# ---------------------------------------------------------
# Exemplo: "tiiuae/falcon-7b-instruct",
MODEL_NAME = "tiiuae/falcon-7b-instruct"
# Verifica se temos GPU disponível
if torch.cuda.is_available():
    device_map_choice = "auto"  # Distribui automaticamente nas GPUs disponíveis
    torch_dtype_choice = torch.float16
else:
    device_map_choice = "cpu"
    torch_dtype_choice = torch.float32

# Carrega o tokenizer
# trust_remote_code=True é necessário para alguns modelos como o Falcon
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)
# Alguns modelos não possuem pad_token; nesse caso, atribuímos o eos_token como pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Carrega o modelo
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map=device_map_choice,
    trust_remote_code=True,
    torch_dtype=torch_dtype_choice
)

# Cria o pipeline de geração
text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.7
)

# Define semente para resultados (relativamente) reproduzíveis
set_seed(42)

# ---------------------------------------------------------
# 2) ESTADO GLOBAL DO CHATBOT
# ---------------------------------------------------------
global_state = {
    "identity": {
        "name": "Viriato",
        "role": "Taberneiro em Viseu",
        "personality": "Acolhedor e ligeiramente brincalhão",
        "goals": ["Ajudar aventureiros", "Proteger a aldeia"]
    },
    "rules": {
        "tone": "Amigável e espirituoso",
        "ethics": "Nunca enganar os jogadores"
    }
}
# Memória de curto prazo (janela de contexto)
context_window = deque(maxlen=20)

# ---------------------------------------------------------
# 3) MEMÓRIA DE LONGO PRAZO (SQLite)
# ---------------------------------------------------------
def initialize_database():
    """
    Cria (se não existir) ou abre a base de dados SQLite
    para armazenar as memórias de longo prazo.
    """
    conn = sqlite3.connect("long_term_memory.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

def save_to_long_term_memory(content: str):
    """
    Guarda texto (conteúdo) na base de dados de longo prazo.
    """
    conn = sqlite3.connect("long_term_memory.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO memories (content) VALUES (?)", (content,))
    conn.commit()
    conn.close()

def query_long_term_memory(query: str) -> List[str]:
    """
    Recupera memórias armazenadas. Exemplo: obter 5 memórias aleatórias.
    (Poderia implementar busca semântica para maior relevância.)
    """
    conn = sqlite3.connect("long_term_memory.db")
    cursor = conn.cursor()
    cursor.execute("SELECT content FROM memories ORDER BY RANDOM() LIMIT 5")
    memories = [row[0] for row in cursor.fetchall()]
    conn.close()
    return memories

# ---------------------------------------------------------
# 4) GERAÇÃO DE RESPOSTAS COM O MODELO LOCAL
# ---------------------------------------------------------
def generate_local_llm_response(prompt: str, max_length: int = 200, temperature: float = 0.7) -> str:
    """
    Gera texto usando o pipeline local (exemplo: Falcon).
    Ajusta 'max_length', 'temperature', etc., conforme necessário.
    """
    try:
        outputs = text_generator(
            prompt,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True
        )
        # O pipeline retorna lista de dicionários, cada qual com 'generated_text'.
        generated_text = outputs[0]["generated_text"]
        return generated_text
    except Exception as e:
        # Caso algo dê errado (ex.: falta de VRAM), retornamos msg de erro
        return f"[ERRO na geração de resposta: {e}]"

def build_system_prompt(global_state, relevant_memories: List[str]) -> str:
    """
    Constrói o prompt de sistema, incluindo identidade, regras e memórias
    relevantes.
    """
    identity_text = (
        f"Tu és {global_state['identity']['name']}, um(a) {global_state['identity']['role']}.\n"
        f"Personalidade: {global_state['identity']['personality']}.\n"
        f"Objetivos: {', '.join(global_state['identity']['goals'])}.\n"
        f"Regras: {global_state['rules']['ethics']}.\n"
    )
    # Se existirem memórias relevantes, adicioná-las
    if relevant_memories:
        memory_text = "Memórias relevantes:\n" + "\n".join(relevant_memories) + "\n"
    else:
        memory_text = "Sem memórias relevantes registradas.\n"
    return identity_text + memory_text

def build_conversation_history(context_window: deque, user_name: str = "Utilizador") -> str:
    """
    Constrói o histórico de conversa a partir da context_window.
    """
    history_str = ""
    for msg in context_window:
        if msg["role"] == "user":
            history_str += f"{user_name}: {msg['content']}\n"
        elif msg["role"] == "assistant":
            history_str += f"Viriato: {msg['content']}\n"
    return history_str

def generate_response(user_input: str, context_window: deque, global_state: dict) -> str:
    """
    Gera a resposta do chatbot:
    1) Obtém memórias relevantes.
    2) Cria o prompt de sistema.
    3) Inclui histórico de mensagens no prompt.
    4) Chama o LLM local.
    5) Faz pós-processamento simples para extrair a parte relevante.
    6) Atualiza memórias de curto e longo prazo.
    """
    # 1) Busca memórias relevantes na base de dados (ex.: 5 aleatórias)
    relevant_memories = query_long_term_memory(user_input)
    # 2) Gera o prompt de sistema
    system_prompt = build_system_prompt(global_state, relevant_memories)
    # 3) Constrói o histórico de conversa
    conversation_history = build_conversation_history(context_window)
    # 4) Cria prompt final para o modelo
    final_prompt = (
        f"{system_prompt}\n"
        f"=== Histórico da conversa ===\n"
        f"{conversation_history}\n"
        f"=== Nova mensagem ===\n"
        f"Utilizador: {user_input}\n"
        f"Viriato:"
    )
    # Gera a resposta crua
    raw_response = generate_local_llm_response(
        final_prompt,
        max_length=400,
        temperature=0.7
    )
    # 5) Extrai apenas a parte depois de "Viriato:" (heurística simples)
    splitted = raw_response.split("Viriato:")
    ai_response = splitted[-1].strip() if len(splitted) > 1 else raw_response
    # 6) Atualiza memória de curto e longo prazo
    context_window.append({"role": "user", "content": user_input})
    context_window.append({"role": "assistant", "content": ai_response})
    # Guarda a interação na memória de longo prazo
    save_to_long_term_memory(f"User: {user_input}")
    save_to_long_term_memory(f"Assistant: {ai_response}")
    return ai_response

# ---------------------------------------------------------
# 5) LOOP PRINCIPAL DO CHAT
# ---------------------------------------------------------
def chatbot():
    print("Viriato: Olá, viajante! Em que posso ser útil hoje?")
    while True:
        user_input = input("Tu: ")
        if user_input.lower() in ["sair", "exit", "quit"]:
            print("Viriato: Adeus, viajante. Boa sorte na tua jornada!")
            break
        response = generate_response(user_input, context_window, global_state)
        print(f"Viriato: {response}")


# ---------------------------------------------------------
# ---------------------------------------------------------
# 6) EXECUTAR O CHATBOT
# ---------------------------------------------------------
if __name__ == "__main__":
    # Inicia/cria a base de dados
    initialize_database()
    # Inicia loop do chatbot
    chatbot()
