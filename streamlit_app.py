from collections import namedtuple
import altair as alt

import os, time
import pandas as pd
import math
import glob
import base64
from io import StringIO

import openai
import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
# -------------IMPORTING CORE FUNCTIONALITIES OF THE SpeeKAR_BOT-------------
from embeddinggenerator import *
from chatbotfunctions import *

# --------------------HTML BUILDER AND FUNCTIONALITIES-----------------------------------#
from htbuilder import (
    HtmlElement,
    div,
    ul,
    li,
    br,
    hr,
    a,
    p,
    img,
    styles,
    classes,
    fonts,
)
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb

import streamlit as st

from PIL import Image


# # ------------------DEFAULTS--------------------#
# os.environ["AZURE_API_KEY"] = 
# os.environ["PINECONE_API_KEY"] = 
api_key =  os.getenv("AZURE_API_KEY")
azure_endpoint = "https://prompt-dashboard.openai.azure.com/"
api_version = "2024-02-15-preview"

# -----------------------HELPER FUNCTIONS--------------------------#
def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):
    style = """
    <style>
      # MainMenu {visibility: display;}
      footer {visibility: display;}
     .stApp { bottom: 105px; }
    </style>
    """
    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 50, 0, 50),
        width=percent(100),
        color="black",
        text_align="left",
        height="auto",
        opacity=1,
    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(1.5),
    )

    body = p()
    foot = div(style=style_div)(hr(style=style_hr), body)

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)


# -------------------------------FUNCTIONS FOR RESPONSE GENERATION-------------#

# def generate_kARanswer(query, text_split):
#     ans, context, keys = chatbot_slim(query, text_split)
#     return ans, context, keys


# -------------------------------------------------------------------------#
# --------------------------GUI CONFIGS------------------------------------#
# -------------------------------------------------------------------------#
# App title
st.set_page_config(page_title="Legal LLaMA")
st.header("Legal LLaMA")


# Hugging Face Credentials
with st.sidebar:
    st.title("Legal LLaMA")
    st.success(
        "Access to this Chatbot is provided by  [AI4BHARAT](https://ai4bharat.iitm.ac.in/)",
        icon="✅",
    )
    hf_email = ""
    hf_pass = "PASS"
    st.markdown(
        "📖 This app is hosted by [AI4BHARAT](https://ai4bharat.iitm.ac.in/)."
    )
    #image = Image.open("speekar_logo.png")
    #st.image(
    #    image,
    #    caption=None,
    #    width=None,
    #    use_column_width=None,
    #    clamp=False,
    #    channels="RGB",
    #    output_format="auto",
    #)


# ---------------------------------------------------------#
# -----------------LOAD THE DOCUMENT INDICES-----------------#
# ---------------------------------------------------------#
st.title("Legal LLaMA: Your virtual assistant for all things legal!")

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How may I assist you today?"}
    ]
    
    
indexName = "llama-chatbot-v1"
# documentsPath = "data/docs_10"
retriever = indexgenerator(indexName)


# ------------------------------------------------------------------------------#
# -------------------------CHATBOT ENGINE TO BE USED -------------#
# ------------------------------------------------------------------------------#

query_engine = context_chatbot_engine(retriever)

# ------------------------------------------------------------------------------#
# -------------------------QUERY AND RESPONSE -------------#
# ------------------------------------------------------------------------------#

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])


# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = query_engine.query(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history





myargs = [
    "Made in India",
    # "" " with ❤️ by ",
    # link("https://github.com/RahulSundar", "@RahulSundar"),
    # br(),
    # link("https://github.com/RahulSundar", "Bhoomi-Nestham-V2.0 @ Gen AI-Chat Bot"),
    # br(),
    # link("https://www.linkedin.com/in/rahul-sundar-311a6977/", "@RahulSundar"),
    # br(),
    # link("https://github.com/RahulSundar", "Bhoomi-Nestham-V2.0 @ Gen AI-Chat Bot"),
]


def footer():
    myargs = [
        "Made in India",
        # "" " with ❤️ by ",
        # link("https://www.linkedin.com/in/rahul-sundar-311a6977/", "@Rahul"),
        # link("https://github.com/RahulSundar", "Bhoomi-Nestham-V2.0 @ Gen AI-Chat Bot"),
    ]
    layout(*myargs)
