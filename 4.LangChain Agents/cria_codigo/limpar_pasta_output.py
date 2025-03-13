import os
import shutil

def limpar_pasta_output(pasta="output"):
    """
    Remove todos os arquivos e subpastas da pasta especificada.
    """
    if os.path.exists(pasta):
        for nome in os.listdir(pasta):
            caminho = os.path.join(pasta, nome)
            try:
                if os.path.isfile(caminho) or os.path.islink(caminho):
                    os.remove(caminho)
                    print(f"Arquivo removido: {caminho}")
                elif os.path.isdir(caminho):
                    shutil.rmtree(caminho)
                    print(f"Pasta removida: {caminho}")
            except Exception as e:
                print(f"Erro ao remover {caminho}: {e}")
        print("Limpeza completa da pasta 'output'.")
    else:
        print("A pasta 'output' não existe.")

if __name__ == "__main__":
    limpar_pasta_output()
