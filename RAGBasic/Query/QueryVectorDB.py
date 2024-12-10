from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
import logging

class QueryVectorDB:
    def __init__(self, chromaPersistenceDirectory, chatModel, embeddingModel):
        self.chromaPersistenceDirectory=chromaPersistenceDirectory
        self.chatModel=chatModel
        self.embeddingModel=embeddingModel
        logging.basicConfig(level=logging.INFO)

    def queryVectorDB(self, question):
        logging.info('Querying the Database for details')
        db = Chroma(persist_directory=self.chromaPersistenceDirectory, embedding_function=self.embeddingModel)
        chain = RetrievalQA.from_chain_type(llm=self.chatModel, retriever=db.as_retriever())
        return chain.invoke({"query": question})
