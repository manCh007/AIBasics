from Utility.LoadToVectorDB import loadToVectorDB
from Utility.QueryVectorDB import queryVectorDB

## For logging purpose
import logging
logging.basicConfig(level=logging.INFO)

def main():
    logging.info('Execution initiated')
    choice = input('If you want to load data and then chat press Y \n')
    if choice=='Y':
         loadToVectorDB()
    while True:
        question = input('Enter your question : ')
        if question == 'exit':
            break
        result = queryVectorDB(question)
        if result['result']:
            print(result['result'])
        else:
            print(result)

if __name__ == '__main__':
    main()