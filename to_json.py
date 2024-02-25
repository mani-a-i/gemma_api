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
        "parameters":{"top_k":3,
                     "top_p":0.9,
                     "temperature":0.9,
                     "max_length":500}  
        
                })[0]['generated_text'][len(text):].strip()





