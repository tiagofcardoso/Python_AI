import requests
from typing import Callable, List
import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM

class RAGChatbot():
    def __init__(self):
        pass

    def carregar_documentos_dos_urls(self, urls: List[str]) -> List[Document]:
        documentos = []
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        for url in urls:
            try:
                loader = PyPDFLoader(url, headers=headers)
                documentos.extend(loader.load())
            except requests.exceptions.RequestException as e:
                print(f"Erro ao carregar o PDF {url}: {e}")
        return documentos

    def dividir_documentos(self, documentos: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        return splitter.split_documents(documentos)

    def criar_vector_store(self, documentos: List[Document], nome_colecao: str = "colecao_documentos") -> Chroma:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = Chroma(
            collection_name=nome_colecao,
            embedding_function=embeddings,
            persist_directory="./chroma_db",
        )

        batch_size = 50
        batches = [documentos[i : i + batch_size] for i in range(0, len(documentos), batch_size)]

        barra_progresso = st.progress(0)
        texto_status = st.empty()
        for indice, batch in enumerate(batches):
            vector_store.add_documents(batch)
            progresso = (indice + 1) / len(batches)

            barra_progresso.progress(progresso)
            texto_status.text(f"A processar: {int(progresso * 100)}% concluído")

        return vector_store

    def obter_retriever(self, vector_store: Chroma, k: int = 10):
        return vector_store.as_retriever(search_kwargs={"k": k})

    def documentos_para_texto(self, docs: List[Document]) -> str:
        return "\n\n".join([doc.page_content for doc in docs])

    def criar_cadeia_rag(
        self, retriever: VectorStoreRetriever, docs_to_string_func: Callable[[List[Document]], str]
    ) -> Runnable:
        template = """
            Responde à pergunta com base apenas no seguinte contexto:
            {context}
            Pergunta: {question}
            Resposta:
        """
        prompt = PromptTemplate.from_template(template)

    def criar_cadeia_rag(self, retriever :VectorStoreRetriever, docs_to_string_func : Callable[[List[Document]], str]) -> Runnable:
        template = """
                Responde à pergunta com base no seguinte context
                {context}
                Pergunta: {question}
                Resposta:
        """
        prompt = PromptTemplate.from_template(template)
        llm = OllamaLLM(model="llama3.2:latest")

        return (
            {"context": retriever | docs_to_string_func, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def perguntar(self, pergunta: str, cadeia_rag: Runnable) -> str:
        return cadeia_rag.invoke(pergunta)

def inicializar_sessao() -> None:
    if "cadeia_rag" not in st.session_state:
        st.session_state.cadeia_rag = None
    if "documento_carregado" not in st.session_state:
        st.session_state.documento_carregado = False
    if "mensagens" not in st.session_state:
        st.session_state.mensagens = []


def carregar_documentos(chatbot: RAGChatbot, urls_pdf: List[str]) -> None:
    if not all(urls_pdf):
        st.warning("Por favor, insere URLs válidos para os PDFs.")
        return

    with st.spinner("A carregar documentos..."):
        documentos = chatbot.carregar_documentos_dos_urls(urls_pdf)
        if not documentos:
            st.error("Falha ao carregar os documentos. Verifica os URLs e tenta novamente.")
            return

        splits = chatbot.dividir_documentos(documentos)
        vector_store = chatbot.criar_vector_store(splits)
        retriever = chatbot.obter_retriever(vector_store)

        st.session_state.cadeia_rag = chatbot.criar_cadeia_rag(retriever, chatbot.documentos_para_texto)
        st.session_state.documento_carregado = True
        st.success(f"Carregados e processados {len(splits)} pedaços de documentos.")
        st.rerun()


if __name__ == '__main__':
    st.title('Chat RAG com documentos pdf')
    chatbot = RAGChatbot()
    inicializar_sessao()

    url_pdf1 = 'https://info.portaldasfinancas.gov.pt/pt/informacao_fiscal/codigos_tributarios/Cod_download/Documents/CIVA.pdf'
    url_pdf2 = 'https://info.portaldasfinancas.gov.pt/pt/informacao_fiscal/codigos_tributarios/Cod_download/Documents/CIRC.pdf'

    if st.button("Carregar Documentos", disabled=st.session_state.documento_carregado):
        carregar_documentos(chatbot, [url_pdf1, url_pdf2])

    if st.session_state.cadeia_rag:
        st.subheader("💬 Conversa com os Documentos")

        if pergunta := st.chat_input("Faz uma pergunta..."):
            resposta = chatbot.perguntar(pergunta, st.session_state.cadeia_rag)
            st.write(resposta)
