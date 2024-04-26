import os, sys
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.memory import ChatMemoryBuffer

from embeddinggenerator import *

# SECRET_API_TOKEN = os.environ["SECRET_API_TOKEN"]
# openai.api_key = SECRET_API_TOKEN

api_key =  os.getenv("AZURE_API_KEY")
azure_endpoint = "https://prompt-dashboard.openai.azure.com/"
api_version = "2024-02-15-preview"

def react_chatbot_engine(index):

    #memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
    chat_engine = index.as_chat_engine(
    chat_mode="react",
    #memory=memory,
    system_prompt=(
        "You are a helpful and friendly chatbot who addresses Land disputes and related grievances, able to have normal interactions, as well as talk"
        " about all the acts, procedures, and solutions related to land disputes in Andhra Pradesh."
        ),
    verbose=True,
    )
    return chat_engine

def condense_question_chatbot_engine(index):

    memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
    chat_engine = index.as_chat_engine(
    chat_mode="condense_question",
    memory=memory,
    system_prompt=(
        "You are a helpful and friendly chatbot who addresses legal queries regarding Indian Law and judgements, able to have normal interactions, as well as talk"
        " about all the acts, judgements and other legal matters in the Indian legal history."
        ),
    verbose=True,
    )
    return chat_engine

def condense_context_question_chatbot_engine(index):

    memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
    chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=memory,
    system_prompt=(
        "You are a helpful and friendly chatbot who addresses legal queries regarding Indian Law and judgements, able to have normal interactions, as well as talk"
        " about all the acts, judgements and other legal matters in the Indian legal history."
        "Here are the relevant documents for the context:\n"
        "{context_str}"
        "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
        ),
    verbose=True,
    )
    return chat_engine


def context_chatbot_engine(retriever):

    memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
    query_engine = RetrieverQueryEngine(retriever=retriever)
    return query_engine


def generate_respone(chat_engine, query):

    response = chat_engine.chat(query)
    return response
    









