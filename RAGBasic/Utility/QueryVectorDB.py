import dotenv
dotenv.load_dotenv()
import os
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma

from Utility.LoadToVectorDB import createEmbeddingModel

## For logging purpose
import logging
logging.basicConfig(level=logging.INFO)

## Creating LLM with Local Models
def createLLM():
    logging.info('Creating the LLM Instance of Ollama')
    llm = Ollama(model=os.getenv('OLLAMA_MODEL_NAME'), temperature=0.0)
    return llm

def queryVectorDB(question):
    logging.info('Querying the Database for details')
    db = Chroma(persist_directory="./chroma_db", embedding_function=createEmbeddingModel())
    chain = RetrievalQA.from_chain_type(llm=createLLM(), retriever=db.as_retriever())
    return chain.invoke({"query": question})