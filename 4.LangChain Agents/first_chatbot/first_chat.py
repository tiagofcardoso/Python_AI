# pip install langserve 
# pip install langchain
# pip install -U langchain-community

from langchain.prompts import PromptTemplate
# from langchain.llms import Ollama
from langchain_ollama.llms import OllamaLLM
from langchain.output_parsers import CommaSeparatedListOutputParser

import uvicorn
from fastapi import FastAPI
from langserve import add_routes

# chain
llama32 = OllamaLLM(model="llama3.2:latest")
prompt =  PromptTemplate.from_template("Conta-me uma piada sobre {topic}")
chain = prompt | llama32 | CommaSeparatedListOutputParser()  # LCEL

# server api
app = FastAPI(title="LangChain Piadas", version=1.0, description="The 1st langserver")
add_routes(app, chain, path="/chain")

if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=8000)

# http://localhost:8000/chain/playground