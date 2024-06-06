import requests


context_path = 'context.txt'
with open(context_path, 'r') as file:
    context = file.read()
            

payload = {
    "context": context,
    "question": "what is for loop?",
    "k": 3,
    "type": 3,
    "chat_history": [],
    "do_spilting": True,
    "add_to_history": True,
    "chunk_size": 400,
    "chunk_overlap":50

}

# TEST CHATBOT 
# response = requests.post(url="http://127.0.0.1:8080/neuarlearn/ml/chat", json=payload)
# print(response.json())

# TEST Summaizer
response = requests.post(url="http://127.0.0.1:8080/neuarlearn/ml/summaizer", json={"text":context, "min_length": 50, "max_length": 250})
print(response.json())


# TEST Transcript

# filename = "Einstein_Surprising.mp3"
# with open(filename, "rb") as f:
#     data = f.read()
  
# response = requests.post(url="http://127.0.0.1:8080/neuarlearn/ml/transcript", data= data)
# print(response.json())
