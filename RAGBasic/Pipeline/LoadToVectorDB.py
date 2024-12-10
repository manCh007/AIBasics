import dotenv
import os
from langchain.text_splitter import TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
import logging

class LoadToVector:
    def __init__(self, embedding, filePath):
        self.embedding=embedding
        self.filePath=filePath
        dotenv.load_dotenv()
        logging.basicConfig(level=logging.INFO)

    ## To load PDF document from the directory
    def convertPdfToDocuments(self):
        logging.info('Convert PDF document to Document')
        # Load files from directory
        files = [self.filePath + '/' + file for file in os.listdir(self.filePath) if file.endswith('.pdf')]
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
    def persistDataToVectorDB(self):
        logging.info('Persisting to Vector DB')
        db = Chroma.from_documents(self.convertPdfToDocuments(), self.embedding, persist_directory="./chroma_db")
        return db