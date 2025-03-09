__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.document_loaders import WebBaseLoader
import streamlit as st

st.title("I am your AI school secretary! How may I help you?")
#add website data

from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings

import os
from getpass import getpass
import chromadb.api
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
# get your free access token from HuggingFace and paste it here
URL = [
"https://www.quora.com/I-want-to-study-computer-engineering-How-do-you-see-specialization-In-which-countries-do-you-advise-me-to-study-so-I-can-find-a-grant",
"https://www.reddit.com/r/ComputerEngineering/comments/1anywuz/need_help_deciding_on_a_specialty_area_for/",
"https://www.quora.com/Which-is-better-normal-computer-science-engineering-and-computer-science-engineering-with-some-specialization?top_ans=286529080",
"https://www.careervillage.org/questions/889126/i-would-like-to-know-the-course-to-study-to-be-a-computer-engineer",
          ]

#load the data
data = WebBaseLoader(URL)
#extract the content
content = data.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=256,chunk_overlap=50)
chunking = text_splitter.split_documents(content)

HF_token = st.text_input("Enter Huggingface Token:", type = "password") #getpass()
clicked = st.button("Submit", key = 1)
if clicked:
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    query = st.text_input("Enter text prompt related to Specializations (Click submit when ready, do not press enter): ")#"What is Bachelor’s Degree in Computer Engineering?"
if clicked and query:
           os.environ['HUGGINGFACEHUB_API_TOKEN'] = HF_token
           embeddings = HuggingFaceInferenceAPIEmbeddings(
           api_key = HF_token,model_name = "BAAI/bge-base-en-v1.5"
           )
           
           
           vectorstore = Chroma.from_documents(chunking, embeddings)
           retriever = vectorstore.as_retriever(search_type="mmr",search_kwargs={"k":3})
           docs_rel = retriever.get_relevant_documents(query)
           #print(docs_rel)
           prompt = f"""
           {query}
           """
           '''
           prompt = f"""
           <|system|>>
           You are an AI Assistant that follows instructions extremely well.
           </s>
           <|user|>
           {query}
           </s>
           <|assistant|>
           """
           '''
           
           
           model = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-alpha",
                        model_kwargs={"temperature":0.5,
                                        "max_new_tokens":512,
                                        "max_length":64
                                        })
           
           qa = RetrievalQA.from_chain_type(llm=model,retriever=retriever,chain_type="stuff")
           response = qa(prompt)
           #print(response['result'])
           st.write(response['result'])
