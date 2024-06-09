import uvicorn
import torch
import os
from fastapi import FastAPI , Request , Depends
from BaseModels.BaseModels import SummarizerIn  ,AudioIn, HumanIn, QuestionGenIN
from summarizer_utils import summarizer_query
from model_utils import Chatbot , Llm_EndPoint
from document_prompts_utils import embedding_model
from question_generation_utils import load_generation_generation_model , question_generation_query
from transcript_utils import query

app = FastAPI()
llm = Llm_EndPoint()
runtime = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = embedding_model(runtime=runtime) 
model , tokenizer = load_generation_generation_model()


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
def summarizer(SummarizerIn:SummarizerIn):
    return summarizer_query(context=SummarizerIn.text, max_length=SummarizerIn.max_length, min_length=SummarizerIn.min_length)


@app.post("/neuarlearn/ml/QuestionGeneration")
def question_generation(QuestionGenIN:QuestionGenIN):

    return question_generation_query(model=model, llm=llm, tokenizer=tokenizer, transcript=QuestionGenIN.transcript, types=QuestionGenIN.types, chunk_size=QuestionGenIN.chunk_size ,chunk_overlap=QuestionGenIN.chunk_overlap) 


if __name__ == "__main__":
    uvicorn.run(app, port=8080, host="127.0.0.1")



