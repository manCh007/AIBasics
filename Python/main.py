### Basic python syntax to use file content ###

# from component.printcomp import PrintComp

# m = PrintComp("Dummy", "Dumbo")
# m.printParam()

###########################################################################################

### Basic Langchain APP to use a query to stream response from local llms ###

# from langchain_community.llms import Ollama

# def main():
#     llm = Ollama(model="llama3")
#     query = "Tell me a joke"

#     for chunks in llm.stream(query):
#         print(chunks)

# if __name__ == "__main__":
#     main()

############################################################################################

### Langchain example to use a prompt template for chatModels using local llms ###

# from langchain_community.chat_models import ChatOllama
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate

# llm = ChatOllama(model="llama3")
# prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")

# chain = prompt | llm | StrOutputParser()

# topic = {"topic": "Space travel"}

# for chunks in chain.stream(topic):
#     print(chunks)


### Langchain prompt templates usnig HumanMessage format ###

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

llm = ChatOllama(model="llama3")
topic = input("Enter your desired topic")
messages = [
    HumanMessage(content="Please tell me about the following topic"),
    HumanMessage(content="{topic}")
]
prompt = ChatPromptTemplate.from_messages(messages)

chain = prompt | llm | StrOutputParser()

param = {"topic": topic}

for chunks in chain.stream(param):
    print(chunks)