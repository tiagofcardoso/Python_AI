import os
import shutil

def limpar_output_exceto_final(pasta="output"):
    """
    Remove todos os arquivos e subpastas da pasta especificada, exceto os que possuem
    'final' no nome.
    """
    if not os.path.exists(pasta):
        print("A pasta 'output' não existe.")
        return

    for nome in os.listdir(pasta):
        caminho = os.path.join(pasta, nome)
        # Se o nome contém 'final', ignora a remoção
        if "final" in nome.lower():
            print(f"Pulando (arquivo/diretório final): {caminho}")
            continue

        try:
            if os.path.isfile(caminho) or os.path.islink(caminho):
                os.remove(caminho)
                print(f"Arquivo removido: {caminho}")
            elif os.path.isdir(caminho):
                shutil.rmtree(caminho)
                print(f"Pasta removida: {caminho}")
        except Exception as e:
            print(f"Erro ao remover {caminho}: {e}")

    print("Limpeza completa da pasta 'output', exceto arquivos finais.")

if __name__ == "__main__":
    limpar_output_exceto_final()
