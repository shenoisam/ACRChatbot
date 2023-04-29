import openai
import os
from dotenv import load_dotenv
load_dotenv()
from LLModel import Model




model_engine='gpt-3.5-turbo'

prompt = "Write a blog on ChatGPT"

# Set the maximum number of tokens to generate in the response
max_tokens = 1024

def main():
    Model().generateIndex()
    query = "How do we eat a potato"
    results = Model().getIndex().similarity_search(query)
    print(results[0].page_content)

if __name__ =="__main__":
    main()