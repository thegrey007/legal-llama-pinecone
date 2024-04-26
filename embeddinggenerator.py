import os, sys
import time
import openai
import logging
import streamlit as st
from llama_index.core import Settings
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader


from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

import os
import openai
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import MetadataReplacementPostProcessor

from pinecone.grpc import PineconeGRPC
from pinecone import ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# os.environ["AZURE_API_KEY"] = 
# os.environ["PINECONE_API_KEY"] = 

api_key =  os.getenv("AZURE_API_KEY")
azure_endpoint = "https://prompt-dashboard.openai.azure.com/"
api_version = "2024-02-15-preview"

llm = AzureOpenAI(
    model="gpt-35-turbo-16k",
    deployment_name="data-gen",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

# You need to deploy your own embedding model as well as your own chat completion model
embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="embed-gen",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

Settings.llm = llm
Settings.embed_model = embed_model

pinecone_api_key = os.getenv("PINECONE_API_KEY")

@st.cache_resource
def indexgenerator(indexname):

    # # check if storage already exists
    # if not os.path.exists(indexPath):
    #     print("Not existing")
    #     # load the documents and create the index
    #     # create the sentence window node parser w/ default settings
    #     node_parser = SentenceWindowNodeParser.from_defaults(
    #         window_size=3,
    #         window_metadata_key="window",
    #         original_text_metadata_key="original_text",
    #     )
    #     # base node parser is a sentence splitter
    #     text_splitter = SentenceSplitter()
    #     Settings.text_splitter = text_splitter
    #     documents = SimpleDirectoryReader(documentsPath).load_data()
    #     print("Part 1\n-------------------------------")
    #     nodes = node_parser.get_nodes_from_documents(documents)
    #     base_nodes = text_splitter.get_nodes_from_documents(documents)
    #     print("Part 2\n-------------------------------")
    #     index = VectorStoreIndex(nodes)
    #     # store it for later
    #     print("Part 3\n-------------------------------")
    #     index.storage_context.persist(persist_dir=indexPath)
    # else:
    # load the existing index
    print("Existing")
    # storage_context = StorageContext.from_defaults(persist_dir=indexname)
    # index = load_index_from_storage(storage_context)
    pc = PineconeGRPC(api_key=pinecone_api_key)
    index_name = "llama-chatbot-v1"
    pinecone_index = pc.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    # Instantiate VectorStoreIndex object from your vector_store object
    vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    # Grab 2 search results
    retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=2)

    return retriever

if __name__ == '__main__':
    indexgenerator('llama-chatbot-v1')
