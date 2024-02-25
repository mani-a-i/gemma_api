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

def translate_customer_message(template_string,customer_style,customer_email):
    chat_pt = ChatPromptTemplate.from_template(template_string)
    customer_message = chat_pt.format_messages(style=customer_style,text=customer_email)[0].content
    print(output(customer_message))


def json_output(template_string,customer_review):
    prompt = ChatPromptTemplate.from_template(template_string).format_messages(text=customer_review)[0].content 
    print(output(prompt))      


template_string = """Translate the text that is delimited by triple backticks into a style that is {style}.\n 
text: ```{text}```
"""
style = """\
a polite tone \
that speaks in English Pirate\
""".strip()

feedback =  """Hey there customer, \
the warranty does not cover \
cleaning expenses for your kitchen \
because it's your fault that \
you misused your blender \
by forgetting to put the lid on before \
starting the blender. \
Tough luck! See ya!
""".strip()

translate_customer_message(template_string,customer_style=style,customer_email=feedback)

json_template = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product \
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

Format the output as JSON with the following keys:
gift
delivery_days
price_value

text: {text}
"""

customer_review = """\
This leaf blower is pretty amazing.  It has four settings:\
candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's \
anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
"""





json_output(template_string=json_template,customer_review=customer_review)



