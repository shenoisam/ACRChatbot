import os
import pickle
from llama_index import download_loader, GPTVectorStoreIndex, ServiceContext,GPTSimpleVectorIndex
from llama_index.gpt_index import  StorageContext,load_index_from_storage
class IndexModel:
    def buildIndex(self):
        documents = SimpleDirectoryReader('data/PracticeGuidelines/total').load_data()
        index = GPTVectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_directory="index")

    def loadIndex(self):
        storage_context = StorageContext.from_defaults(persist_dir="./index")
        # load index
        index = load_index_from_storage(storage_context)
        return index
    def run (self,prompt):
        index = self.loadIndex()
        response = index.query(prompt)

        # Get the last token usage
        last_token_usage = index.llm_predictor.last_token_usage
        print(f"last_token_usage={last_token_usage}")
        return response