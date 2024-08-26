import langchain
#from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain_community.chains.pebblo_retrieval.base import PebbloRetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain import llms, chains
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
import openai
#from langchain_community.llms import Openai

from langchain_community.vectorstores import Chroma

from langchain.text_splitter import CharacterTextSplitter

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader

from langchain.indexes import VectorstoreIndexCreator
from langchain_openai import OpenAI
from langchain_community.llms import OpenAI


import os
import openai
import gradio as gr
import os  

os.environ['OPENAI_API_KEY'] = ''
def get_answer(query):
    file_path = 'info.txt'
    if not os.path.exists(file_path):
        raise RuntimeError(f"File {file_path} does not exist.")
    
    try:
        loader = TextLoader(file_path)
        document = loader.load()
    except Exception as e:
        raise RuntimeError(f"Error loading {file_path}") from e

    text_spliter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_spliter.split_documents(document)

    embedding = OpenAIEmbeddings(openai_api_key=openai.api_key)
    docsearch = Chroma.from_documents(texts, embedding)

    qa = langchain.chains.VectorDBQA.from_chain_type(llm=langchain.llms.OpenAI(), chain_type="stuff", vectorstore=docsearch,
                                                     return_source_documents=True)

    result = qa({'query': query})
    return result['result']
