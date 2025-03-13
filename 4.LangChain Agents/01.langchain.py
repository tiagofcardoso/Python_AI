#pip install langserve
#pip install langchain
import ollama
import langchain
from langchain.prompts import PromptTemplate
from langchain.llms import ollama
from langchain.output_parsers import CommaSeparatedListOutputParser

llama3 = ollama(model="llama3:latest")
prompt = PromptTemplate.from_template("Diga algo engraçado {topic}")
chain = prompt | llama3 | CommaSeparatedListOutputParser()

