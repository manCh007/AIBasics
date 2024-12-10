import os
import logging
import dotenv
from pydantic import BaseModel, Field
from langchain.text_splitter import TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.graph_transformers.llm import SystemMessage
from langchain_experimental.graph_transformers.llm import JsonOutputParser
from langchain_experimental.graph_transformers.llm import PromptTemplate
from langchain_experimental.graph_transformers.llm import HumanMessagePromptTemplate
from langchain_experimental.graph_transformers.llm import ChatPromptTemplate
from langchain_experimental.graph_transformers import LLMGraphTransformer


class UnstructuredRelation(BaseModel):
    head: str = Field(
        description=(
            "extracted head entity like Merchant, Address, City, State etc."
            "Must use human-readable unique identifier."
        )
    )
    head_type: str = Field(
        description="type of the extracted head entity like Merchant, Address, City, State etc"
    )
    relation: str = Field(description="relation between the head and the tail entities")
    tail: str = Field(
        description=(
            "extracted head entity like Merchant, Address, City, State etc."
            "Must use human-readable unique identifier."
        )
    )
    tail_type: str = Field(
        description="type of the extracted head entity like Merchant, Address, City, State etc"
    )


def buildChatPrompt():
    logging.info('Creating a chat_prompt to provide the LLM with the instructions and examples')
    system_prompt = """
    You are a data scientist working for the police and you are building a knowledge graph database. 
    Your task is to extract information from data and convert it into a knowledge graph database.
    Provide a set of Nodes in the form [head, head_type, relation, tail, tail_type].
    It is important that the head and tail exists as nodes that are related by the relation.
    If you can't pair a relationship with a pair of nodes don't add it.
    When you find a node or relationship you want to add try to create a generic TYPE for it that describes the entity you can also think of it as a label.
    You must generate the output in a JSON format containing a list with JSON objects. Each object should have the keys: "head", "head_type", "relation", "tail", and "tail_type".
    """
    system_message = SystemMessage(content=system_prompt)
    parser = JsonOutputParser(pydantic_object=UnstructuredRelation)

    examples = [
        {
            "text": (
                "Merchant: Punjab Crockery; Address: HyderabadGROUND FLOOR,WHITE HOUSE, BEGUMPET; City: HYDERABAD, State: Andhra Pradesh"
            ),
            "head": "Punjab Crockery",
            "head_type": "Merchant",
            "relation": "HAS_ADDRESS",
            "tail": "HyderabadGROUND FLOOR,WHITE HOUSE, BEGUMPET",
            "tail_type": "Address",
        },
        {
            "text": (
                "Merchant: Punjab Crockery; Address: HyderabadGROUND FLOOR,WHITE HOUSE, BEGUMPET; City: HYDERABAD, State: Andhra Pradesh"
            ),
            "head": "Punjab Crockery",
            "head_type": "Merchant",
            "relation": "FROM_CITY",
            "tail": "HYDERABAD",
            "tail_type": "City",
        },
        {
            "text": (
                "Merchant: Punjab Crockery; Address: HyderabadGROUND FLOOR,WHITE HOUSE, BEGUMPET; City: HYDERABAD, State: Andhra Pradesh"
            ),
            "head": "Punjab Crockery",
            "head_type": "Merchant",
            "relation": "FROM_STATE",
            "tail": "Andhra Pradesh",
            "tail_type": "State",
        },
        {
            "text": (
                "Punjab Crockery HyderabadGROUND FLOOR,WHITE HOUSE, BEGUMPET HYDERABAD Andhra Pradesh"
            ),
            "head": "Punjab Crockery",
            "head_type": "Merchant",
            "relation": "HAS_ADDRESS",
            "tail": "HyderabadGROUND FLOOR,WHITE HOUSE, BEGUMPET",
            "tail_type": "Address",
        },
        {
            "text": (
                "Punjab Crockery HyderabadGROUND FLOOR,WHITE HOUSE, BEGUMPET HYDERABAD Andhra Pradesh"
            ),
            "head": "Punjab Crockery",
            "head_type": "Merchant",
            "relation": "FROM_CITY",
            "tail": "HYDERABAD",
            "tail_type": "City",
        },
        {
            "text": (
                "Punjab Crockery HyderabadGROUND FLOOR,WHITE HOUSE, BEGUMPET HYDERABAD Andhra Pradesh"
            ),
            "head": "Punjab Crockery",
            "head_type": "Merchant",
            "relation": "FROM_STATE",
            "tail": "Andhra Pradesh",
            "tail_type": "State",
        },
        {
            "text": (
                "01 Punjab Crockery HyderabadGROUND FLOOR,WHITE HOUSE, BEGUMPET HYDERABAD Andhra Pradesh"
            ),
            "head": "Punjab Crockery",
            "head_type": "Merchant",
            "relation": "HAS_ADDRESS",
            "tail": "HyderabadGROUND FLOOR,WHITE HOUSE, BEGUMPET",
            "tail_type": "Address",
        },
        {
            "text": (
                "01 Punjab Crockery HyderabadGROUND FLOOR,WHITE HOUSE, BEGUMPET HYDERABAD Andhra Pradesh"
            ),
            "head": "Punjab Crockery",
            "head_type": "Merchant",
            "relation": "FROM_CITY",
            "tail": "HYDERABAD",
            "tail_type": "City",
        },
        {
            "text": (
                "01 Punjab Crockery HyderabadGROUND FLOOR,WHITE HOUSE, BEGUMPET HYDERABAD Andhra Pradesh"
            ),
            "head": "Punjab Crockery",
            "head_type": "Merchant",
            "relation": "FROM_STATE",
            "tail": "Andhra Pradesh",
            "tail_type": "State",
        }
    ]
    human_prompt = PromptTemplate(
        template="""
    Examples:
    {examples}

    For the following text, extract entities and relations as in the provided example.
    {format_instructions}\nText: {input}""",
        input_variables=["input"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
            "node_labels": None,
            "rel_types": None,
            "examples": examples,
        },
    )
    human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message, human_message_prompt]
    )
    return chat_prompt


class LoadToKG:
    def __init__(self, filePath, chatModel):
        self.filePath=filePath
        self.chatModel=chatModel
        dotenv.load_dotenv()
        logging.basicConfig(level=logging.INFO)

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

    def buildGraphDocument(self):
        logging.info(
            'Instantiating the LLMGraphTransformer that will extract the entities and relationships from the Documents')
        llm_transformer = LLMGraphTransformer(llm=self.chatModel, prompt=buildChatPrompt())

        logging.info('Converting the Documents into Graph Documents...')
        graph_documents = llm_transformer.convert_to_graph_documents(self.convertPdfToDocuments())

        return graph_documents

    def loadGraphDB(self):
        logging.info('Instantiating the Neo4JGraph to persist the data')
        from langchain_community.graphs import Neo4jGraph
        graph = Neo4jGraph()

        logging.info('Persisting the Graph Documents into the Neo4JGraph')
        graph.add_graph_documents(
            self.buildGraphDocument(),
            baseEntityLabel=True,
            include_source=True
        )

        logging.info('Data pipeline completed successfully!')
