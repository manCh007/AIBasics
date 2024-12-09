import dotenv
dotenv.load_dotenv()
import os
from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

## For logging purpose
import logging
logging.basicConfig(level=logging.INFO)

## Creating Embedding Model
def createEmbeddingModel():
    logging.info('Creating Embedding Model of Ollama')
    embedding = OllamaEmbeddings(model=os.getenv('OLLAMA_EMBEDDING_MODEL'))
    return embedding

## To load PDF document from the directory
def convertPdfToDocuments():
    logging.info('Convert PDF document to Document')
    # Load files from directory
    files_path = 'Data'
    files = [files_path+'/'+file for file in os.listdir(files_path) if file.endswith('.pdf')]
    # Create text splitter
    splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
    documents = []
    # Create documents out of file
    for file in files:
        pdf_loader = PyPDFLoader(file_path=file)
        files_document = pdf_loader.load_and_split(text_splitter=splitter)
        documents.extend(files_document)
    # return documents
    return documents

## Load to vector database
def persistDataToVectorDB(documents, embedding):
    logging.info('Persisting to Vector DB')
    db = Chroma.from_documents(documents, embedding, persist_directory="./chroma_db")
    return db

## To load the data from the files and load to Vector Database
def loadToVectorDB():
    logging.info('Loading to Vector DB - Series of action initiated')
    documents = convertPdfToDocuments()
    db = persistDataToVectorDB(documents, createEmbeddingModel())
    return db