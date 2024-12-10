import logging
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM

class LLMUtility:
    def __init__(self, embeddingModelName, chatModelName):
        self.embeddingModelName=embeddingModelName
        self.chatModelName=chatModelName
        logging.basicConfig(level=logging.INFO)

    def createEmbeddingModel(self):
        logging.info('Creating Embedding Model of Ollama')
        embedding = OllamaEmbeddings(model=self.embeddingModelName)
        return embedding

    def createLLM(self):
        logging.info('Creating the LLM Instance of Ollama')
        llm = OllamaLLM(model=self.chatModelName, temperature=0.0)
        return llm