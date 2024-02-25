import os
import torch
import requests
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
from langchain.prompts import ChatPromptTemplate


def config():
    load_dotenv()
config()

API_URL = "https://api-inference.huggingface.co/models/google/gemma-7b-it"
headers = {"Authorization": "Bearer {}".format(os.getenv('API_KEY'))}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)    
    return response.json()

def gemma_template(prompt):
    text =  "<start_of_turn>user\n{}<end_of_turn>".format(prompt) + "\n<start_of_turn>model"    
    return text

def output(prompt):
    text =  gemma_template(prompt)   
    
    return query({
        "inputs":text,
        "parameters":{"top_k":2,
                     "top_p":0.9,
                     "temperature":0.5,
                     "max_length":500}  
        
    })[0]['generated_text'][len(text):].strip()

def translate_customer_message(template_string,customer_style,customer_email):
    chat_pt = ChatPromptTemplate.from_template(template_string)
    customer_message = chat_pt.format_messages(style=customer_style,text=customer_email)[0].content
    print(output(customer_message))

    
template_string = """Translate the text that is delimited by triple backticks into a style that is {style}.\n 
text: ```{text}```
"""
customer_style = """American English 
in a calm and respectful tone
""".strip()

customer_email = """Arrr, I be fuming that me blender lid flew off and splattered me kitchen walls with smoothie! And to make matters worse, the warranty don't cover the cost of cleaning up me kitchen. I need yer help right now, matey!""".strip()


translate_customer_message(template_string,customer_style,customer_email)


