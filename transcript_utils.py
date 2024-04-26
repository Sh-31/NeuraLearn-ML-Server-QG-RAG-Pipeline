import os
from dotenv import load_dotenv 
import requests

load_dotenv()

def transcribe_audio_to_text_file(model,file_path):
    model = model.load_model("small")
    result = model.transcribe(audio=file_path)
    segments = result['segments']
    file_name = os.path.basename(file_path)

    # Ensure the output directory exists
    output_directory = ""
    os.makedirs(output_directory, exist_ok=True)

    text_filename = os.path.join(output_directory, f"{file_name}.txt")

    for segment in segments:
        text = segment['text']

        with open(text_filename, 'a', encoding='utf-8') as text_file:
            text_file.write(f"{text}\n")

    return text_filename



API_TOKN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-medium.en"
headers = {"Authorization": "Bearer {}".format(API_TOKN)}

def query(data):
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()