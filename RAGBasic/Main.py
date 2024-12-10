import os
import dotenv
from Utility.LLMUtility import LLMUtility
from Pipeline.LoadToVectorDB import LoadToVector
from Query.QueryVectorDB import QueryVectorDB
from Pipeline.LoadToKG import LoadToKG

dotenv.load_dotenv()

import logging
logging.basicConfig(level=logging.INFO)

def main():
    logging.info('Execution initiated')
    llmUtil = LLMUtility(embeddingModelName=os.getenv('OLLAMA_EMBEDDING_MODEL'),
                         chatModelName=os.getenv('OLLAMA_MODEL_NAME'))
    ifGraphRAG = input('Do you want to use Graph RAG')
    if not ifGraphRAG:
        choice = input('If you want to load data and then chat press Y \n')
        if choice=='Y':
            LoadToVector(embedding=llmUtil.createEmbeddingModel(), filePath='Data').persistDataToVectorDB()
        while True:
            question = input('Enter your question : ')
            if question == 'exit':
                break
            result = (QueryVectorDB(chromaPersistenceDirectory='./chroma_db',
                                    chatModel=llmUtil.createLLM(),
                                    embeddingModel=llmUtil.createEmbeddingModel())
                      .queryVectorDB(question))
            if result['result']:
                print(result['result'])
            else:
                print(result)
    choice = input('If you want to load data and then chat press Y \n')
    if choice == 'Y':
        LoadToKG(filePath='Data', chatModel=llmUtil.createLLM()).loadGraphDB()

if __name__ == '__main__':
    main()