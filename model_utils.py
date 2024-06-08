import os
from dotenv import load_dotenv 
from langchain_community.llms import HuggingFaceEndpoint # type: ignore
from langchain_core.runnables import RunnablePassthrough , RunnableParallel
from document_prompts_utils import prompt , text_splitter
from semantic_similarity_utils import vector_db as vector_db_
from memory_utils import chat

def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

def get_completion(prompt:str, llm):
    return llm.invoke(prompt)


def GemmaLLM_Api_EndPoint():
    llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.3",
                          max_new_tokens=1000,
                          huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKN"),
                          )
    return llm

def invoke_with_source(llm, db, query:str, k:int=3):
    
    rag_chain = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["document"])))
    | prompt
    | llm
    )

    rag_chain_with_source = RunnableParallel(
        {"document":  db.as_retriever(search_type="similarity", search_kwargs={"k": k})  , "query": RunnablePassthrough()}
    ).assign(answer=rag_chain)

    return rag_chain_with_source.invoke(query)



def Chatbot(llm, embedding_model, question, context="",
            search_type="similarity", vector_db=None, k=3, do_spilting=True, type:int = 1, 
            chunk_size:int=2000, chunk_overlap:int=200 ,
            separators:list[str] =["\n\n", "\n", " ", ""], add_to_history=True, chat_history=[]):

    if context != "" and vector_db == None:
         
         if do_spilting:
            text_splitte = text_splitter(text=context,type=type,chunk_size=chunk_size,chunk_overlap=chunk_overlap,separators=separators)
         else :  
            text_splitte = [context]

         db = vector_db_(text=text_splitte,embdedding=embedding_model)

         vector_db = db
       
    retrieval_db = db.as_retriever(search_type=search_type, 
                               search_kwargs={"k": k})
    
    result, chat_history = chat(llm, question=question,retriever=retrieval_db,
         chat_history=chat_history, add_to_history=add_to_history)
    

    return result , chat_history , vector_db



def load_model():
    # import torch  # type: ignore
    # from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig , pipeline
    # from langchain_community.embeddings import HuggingFaceEmbeddings # type: ignore
    # from langchain.llms import HuggingFacePipeline # type: ignore
    # from langchain.embeddings.huggingface import HuggingFaceEmbeddings # type: ignore
        # Loud our fine-tuned model as HuggfacePipline model
    # model_name = "shredder-31/GA_model_Gemma_2b"
    # bnb_config = BitsAndBytesConfig(load_in_4bit=True, 
    #                                 bnb_4bit_quant_type="nf4", 
    #                                 bnb_4bit_compute_dtype=torch.float16,
    #                                 bnb_4bit_use_double_quant=True
    #                             )



    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     quantization_config=bnb_config,
    # )

    # tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True)

    # text_generation_pipeline = pipeline(
    #     model=model,
    #     tokenizer=tokenizer,
    #     task="text-generation",
    #     repetition_penalty=1.1,
    #     return_full_text=True,
    #     max_new_tokens=2048,
    #     output_scores=True
    # )

    # GemmaLLM = HuggingFacePipeline(pipeline=text_generation_pipeline)
    pass 