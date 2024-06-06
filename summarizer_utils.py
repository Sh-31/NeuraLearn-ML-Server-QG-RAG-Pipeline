import os
from dotenv import load_dotenv 
import requests

load_dotenv()

API_TOKN = os.getenv('HUGGINGFACEHUB_API_TOKEN')


API_URL = "https://api-inference.huggingface.co/models/shredder-31/Summarization_Model_led_base_book_summary"

headers = {"Authorization": "Bearer {}".format(API_TOKN) }

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

# from langchain_community.llms import HuggingFaceEndpoint 
#  summarizer = HuggingFaceEndpoint(repo_id="shredder-31/Summarization_Model_led_base_book_summary",
#                           task="summarization",           
#                           max_new_tokens=250,
#                           huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKN"),
#                           )


def summarizer_query(context, max_length=250, min_length=50):
    return query({
	"inputs": context,
    "max_length":max_length,
    "min_length":min_length,
    "return_full_text":True,
})