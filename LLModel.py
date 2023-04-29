# Author: Sam Shenoi
# Description: This file builds a class for interfacing with ChatGPT


from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
class Model:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.9,model_name='gpt-3.5-turbo')

    def generatePrompt(self):
        prompt = PromptTemplate(
            input_variables=["scenario"],
            template="What imaging based on ACR guidelines is recommended for this clinical scenario {scenario}?"
        )
        return prompt

    def generateIndex(self,filename="data/PracticeGuidelines/largefile.txt"):
        loader = TextLoader(filename)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        rds = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory="index")
        rds.persist()
        rds = None
        return rds

    def getIndex(self):
        embeddings = OpenAIEmbeddings()
        rds = Chroma(persist_directory="index", embedding_function=embeddings)
        return rds

    def test(self,index):
        qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=index.as_retriever())
        
    def run(self):
        prompt = self.generatePrompt()
        chain = LLMChain(llm=self.llm, prompt=prompt)

