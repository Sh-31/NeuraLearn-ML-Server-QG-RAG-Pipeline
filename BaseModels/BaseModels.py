from pydantic import BaseModel
from typing import Optional , Union


class SummarizerIn(BaseModel):
    text: str
    max_length: Optional[int]
    min_length: Optional[int]

class AudioIn(BaseModel):
    audio: bytes 

class HumanIn(BaseModel):
    question: str
    context: Optional[str]
    k: Optional[int]
    type: Optional[int]
    chat_history: Optional[list[str]]    
    do_spilting: Optional[bool]
    add_to_history: Optional[bool]
    chunk_size: Optional[int]
    chunk_overlap: Optional[int]

class LlmOut(BaseModel):
    answer: str
    chat_history: Optional[list[str]]
    