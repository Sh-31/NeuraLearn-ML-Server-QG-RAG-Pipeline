import uvicorn
import torch
from fastapi import FastAPI , Request , Depends
from BaseModels.BaseModels import TextIn  ,AudioIn, HumanIn
from summarizer_utils import summarizer_query
from model_utils import Chatbot , GemmaLLM_Api_EndPoint
from document_prompts_utils import embedding_model
from transcript_utils import query

app = FastAPI()
llm = GemmaLLM_Api_EndPoint()
runtime = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = embedding_model(runtime=runtime) 


async def parse_body(request: Request):
    AudioIn: bytes = await request.body()
    return AudioIn


@app.post("/neuarlearn/ml/transcript")
async def transcript(AudioIn: bytes = Depends(parse_body)):
    transcript = query(AudioIn)
    return transcript


@app.post("/neuarlearn/ml/chat")
def question_answer(HumanIn:HumanIn):
    
    result , chat_history , vecter_db = Chatbot(llm=llm,
                                                embedding_model=embed_model, 
                                                context=HumanIn.context, 
                                                chat_history=HumanIn.chat_history,
                                                question=HumanIn.question)
            
    return {"answer":result["answer"], "chat_history" : chat_history}


@app.post("/neuarlearn/ml/summaizer")
def summarizer(text:TextIn):
    return summarizer_query(context=text.text)



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)



