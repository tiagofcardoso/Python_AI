import os
import json
from datetime import datetime
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate


class CodeEvolution():
    def __init__(self, tarefa, nome_base, linguagem_programacao="Python", extensao=".py", output_dir="output", model="llama3:latest", iterations=5):
        """
        Inicializa a classe CodeEvolution com os parâmetros da tarefa.
        """
        self.tarefa = tarefa
        self.nome_base = nome_base
        self.linguagem_programacao = linguagem_programacao
        self.extensao = extensao
        self.output_dir = output_dir
        self.iterations = iterations

        os.makedirs(self.output_dir, exist_ok=True)
        self.ollama_llm = OllamaLLM(model=model)

        self.prompt_inicial_template = PromptTemplate(
            input_variables=["linguagem", "tarefa"],
            template=(
                "És um programador especialista. A tua tarefa é escrever código em {linguagem} para {tarefa}.\n"
                "Por favor, gera o código em {linguagem} e retorne APENAS o código, sem explicações:"
            )
        )

        self.prompt_melhoria_template = PromptTemplate(
            input_variables=["linguagem", "tarefa", "codigo_anterior"],
            template=(
                "És um programador especialista. A tua tarefa é escrever código em {linguagem} para {tarefa}.\n"
                "Aqui está o código anterior que geraste:\n"
                "{codigo_anterior}\n"
                "Por favor, melhora este código. Torna-o mais profissional, mais eficiente, legível e corrige quaisquer erros.\n"
                "Gera o código {linguagem} melhorado e retorne APENAS o código, sem explicações:"
            )
        )

        self.codigo_inicial_chain = self.prompt_inicial_template | self.ollama_llm
        self.codigo_melhoria_chain = self.prompt_melhoria_template | self.ollama_llm

    def extract_valid_code(self, text):
        """
        Extrai o código válido removendo delimitadores de bloco.
        """
        lines = text.splitlines()
        code_lines = []
        in_code_block = False
        language_delimiter = "```" + self.linguagem_programacao.lower()
        for line in lines:
            if not in_code_block and line.strip().lower().startswith(language_delimiter):
                in_code_block = True
                continue
            if in_code_block:
                if line.strip() == "```":
                    break
                code_lines.append(line)
        if code_lines:
            return "\n".join(code_lines).strip()
        return text.strip()

    def run(self):
        """
        Executa a geração e melhoria do código por um número definido de iterações.
        """
        nome_ficheiro_log = f"{self.nome_base}_evolucao.txt"
        log_file_path = os.path.join(self.output_dir, nome_ficheiro_log)
        codigo_anterior_iteracao = ""

        with open(log_file_path, "w", encoding="utf-8") as log_file:
            log_file.write(
                f"Log de Evolução de Código - Tarefa: {self.tarefa}, Linguagem: {self.linguagem_programacao}\n"
            )
            log_file.write(
                f"Timestamp inicial: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            for i in range(self.iterations):
                print(f"Iteração {i+1} para {self.nome_base}:")
                log_file.write(f"Iteração {i+1}:\n")

                if i == 0:
                    input_dict = {
                        "linguagem": self.linguagem_programacao, "tarefa": self.tarefa}
                    codigo_gerado = self.codigo_inicial_chain.invoke(
                        input_dict)
                else:
                    input_dict = {
                        "linguagem": self.linguagem_programacao,
                        "tarefa": self.tarefa,
                        "codigo_anterior": codigo_anterior_iteracao,
                    }
                    codigo_gerado = self.codigo_melhoria_chain.invoke(
                        input_dict)

                print(
                    f"Código Gerado na Iteração {i+1} para {self.nome_base}:\n{codigo_gerado}")
                log_file.write(
                    f"Código Gerado na Iteração {i+1}:\n{codigo_gerado}\n")

                codigo_anterior_iteracao = codigo_gerado
                codigo_limpo = self.extract_valid_code(codigo_gerado)

                nome_ficheiro_iteracao = f"{self.nome_base}_i{i+1}{self.extensao}"
                if i == self.iterations - 1:
                    nome_ficheiro_iteracao = f"{self.nome_base}_i{self.iterations}_final{self.extensao}"

                ficheiro_codigo_path = os.path.join(
                    self.output_dir, nome_ficheiro_iteracao)
                with open(ficheiro_codigo_path, "w", encoding="utf-8") as f:
                    f.write(codigo_limpo)

                print(
                    f"Código da Iteração {i+1} para {self.nome_base} guardado em: {ficheiro_codigo_path}")
                log_file.write(
                    f"Código da Iteração {i+1} guardado em: {ficheiro_codigo_path}\n\n")

            print(
                f"\nCódigo Final Melhorado para {self.nome_base} após {self.iterations} iterações:")
            print(codigo_anterior_iteracao)
            log_file.write(
                f"\nCódigo Final Melhorado:\n{codigo_anterior_iteracao}\n")
            log_file.write(
                f"Timestamp final: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        print(
            f"\nEvolução completa e código final para {self.nome_base} foram guardados no ficheiro de log: {log_file_path}")
        return log_file_path

    def clean_backticks(self):
        """
        Realiza a limpeza dos arquivos gerados removendo os delimitadores de código (backticks).
        """
        for filename in os.listdir(self.output_dir):
            if filename.endswith(self.extensao):
                ficheiro_path = os.path.join(self.output_dir, filename)
                with open(ficheiro_path, "r", encoding="utf-8") as f:
                    conteudo = f.read()
                conteudo_limpo = conteudo.replace("```", "")
                with open(ficheiro_path, "w", encoding="utf-8") as f:
                    f.write(conteudo_limpo)
                print(f"Limpeza de backticks realizada em: {ficheiro_path}")

if __name__ == '__main__':
    try:
        with open("tasks.json", "r", encoding="utf-8") as f:
            tasks = json.load(f)
    except Exception as e:
        print(f"Erro ao ler o ficheiro de tarefas: {e}")
        exit(1)

    # Processa cada tarefa lida do ficheiro
    for task in tasks:
        evolucao = CodeEvolution(
            tarefa=task["tarefa"],
            nome_base=task["nome_base"],
            linguagem_programacao=task["linguagem"],
            extensao=task["extensao"]
        )
        evolucao.run()
        evolucao.clean_backticks()
