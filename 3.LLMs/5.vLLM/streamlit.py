import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Título da aplicação
st.title("Aplicativo de Geração de Texto com LLM Fine Tuned")

# Carregar o modelo e o tokenizer
# Certifique-se de que o modelo fine tuned esteja salvo no diretório informado
model_path = "./modelo_fine_tuned"  # Altere este caminho conforme necessário

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
except Exception as e:
    st.error(f"Erro ao carregar o modelo: {e}")
    st.stop()

# Área para inserção do prompt
st.subheader("Entrada de Prompt")
prompt = st.text_area("Digite seu prompt aqui:")

# Slider para definir o tamanho máximo da geração (em tokens)
max_length = st.slider("Tamanho máximo da geração (em tokens):",
                       min_value=50, max_value=300, value=100)

# Botão para acionar a geração de texto
if st.button("Gerar Texto"):
    if prompt.strip() == "":
        st.warning("Por favor, insira um prompt válido!")
    else:
        with st.spinner("Gerando texto..."):
            # Parâmetros de geração podem ser ajustados conforme a necessidade
            outputs = generator(prompt, max_length=max_length,
                                do_sample=True, top_k=50, top_p=0.95)
            texto_gerado = outputs[0]["generated_text"]
        st.subheader("Texto Gerado")
        st.write(texto_gerado)
