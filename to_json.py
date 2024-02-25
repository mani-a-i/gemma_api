import os
import torch
import requests
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser



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
        "parameters":{"top_k":1,
                     "top_p":0.9,
                     "temperature":0.98,
                     "max_length":500}  
        
                })[0]['generated_text'][len(text):].strip()




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

review_template_2 = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product\
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

text: {text}

{format_instructions}
"""


gift_schema = ResponseSchema(name="gift",
                             description="Was the item purchased\
                             as a gift for someone else? \
                             Answer True if yes,\
                             False if not or unknown.")
delivery_days_schema = ResponseSchema(name="delivery_days",
                                      description="How many days\
                                      did it take for the product\
                                      to arrive? If this \
                                      information is not found,\
                                      output -1.")
price_value_schema = ResponseSchema(name="price_value",
                                    description="Extract any\
                                    sentences about the value or \
                                    price, and output them as a \
                                    comma separated Python list.")

response_schemas = [gift_schema, 
                    delivery_days_schema,
                    price_value_schema]


def json_out(response_schemas,review_template_2,customer_review):

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_template(template=review_template_2)

    messages = prompt.format_messages(text=customer_review, 
                                      format_instructions=format_instructions)[0].content

    print(output(messages))


json_out(response_schemas,review_template_2,customer_review)