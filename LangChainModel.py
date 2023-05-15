# Author: Sam Shenoi
# Description: This file builds a class for interfacing with ChatGPT


from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader,DirectoryLoader, PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import csv
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain, PromptTemplate

class Model:
    def __init__(self):
        self.llm = OpenAI(temperature=0)
        self.rds = None
        self.embeddings  = OpenAIEmbeddings()

    def generatePrompt(self):
        prompt = PromptTemplate(
            input_variables=["scenario"],
            template="What imaging based on ACR guidelines is recommended for this clinical scenario {scenario}?"
        )
        return prompt

    def generateIndex(self,indexname, filename="data/acrguidelines/txt/temp"):
        loader = DirectoryLoader(filename, glob="*.txt")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        rds = Chroma.from_documents(documents=docs, embedding=self.embeddings, persist_directory=indexname)
        rds.persist()
        return rds

    def moreIntelligentIndex(self,indexname, filename="data/acrguidelines/txt"):
        loader = DirectoryLoader(filename, glob="*.txt")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        data = []
        with open('out.tsv', newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spamreader:
                print(row)
                data.append((row[0],row[1]))

        for d in docs[0:10]:
            condition = None
            for i in data:
                if i[0] == d.metadata["source"]:
                    condition = i[1]
                    break
            d.metadata["condition"] = condition
        vectorstore = Chroma.from_documents(documents=docs, embedding=self.embeddings, persist_directory=indexname)
        return vectorstore
    def self_query(self,vectorstore,query):
        metadata_field_info=[
            AttributeInfo(
                name="condition",
                description="A particular clinical condition",
                type="string"
            ),
        ]
        document_content_description = "ACR imaging recommendations"
        retriever = SelfQueryRetriever.from_llm(self.llm, vectorstore, document_content_description, metadata_field_info, verbose=True)
        return retriever.get_relevant_documents(query)
    def getIndex(self,index):
        if self.rds is None:
            rds = Chroma(persist_directory=index, embedding_function=self.embeddings)
            self.rds = rds
        return self.rds

    def test(self,query,index="indexComplete"):
        index = self.getIndex(index)
        qa = RetrievalQA.from_chain_type(llm=self.llm, chain_type="refine", retriever=index.as_retriever())
        print(qa.run(query))


    def run(self,query,index="indexComplete"):
        index = self.getIndex(index)
        question_prompt_template = """You are a radiology support bot. Use the following portion of a long document to see if any of the text applies to the provided clinical case.
            {context}
            Question: {question}
            Relevant text, if any:"""
        QUESTION_PROMPT = PromptTemplate(template=question_prompt_template, input_variables=["context", "question"])

        combine_prompt_template = """You are a radiology support bot. Given the following extracted parts of a long document and a clinical case, create a final answer indicating what type, if any, of imaging is indicated.
        If you don't know the type of imaging, just say that you don't know. Don't try to make up an answer.

        QUESTION: {question}
        =========
        {summaries}
        =========
        Answer:"""
        COMBINE_PROMPT = PromptTemplate(
            template=combine_prompt_template, input_variables=["summaries", "question"]
        )
        chain = load_qa_chain(self.llm, chain_type="map_reduce", return_map_steps=True, question_prompt=QUESTION_PROMPT, combine_prompt=COMBINE_PROMPT)

        #chain = load_qa_chain(OpenAI(temperature=0), chain_type="map_reduce",prompt=prompt_template)
        docs = index.as_retriever().get_relevant_documents(query)
        result = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
        return  result["output_text"]


