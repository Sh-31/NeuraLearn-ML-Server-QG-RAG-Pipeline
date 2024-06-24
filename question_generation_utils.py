from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from parsers_utils import parse_text
from document_prompts_utils import text_splitter
import numpy as np
import torch

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Input:
{}

### Response:
{}"""

def clean_transcript(transcript:str,llm):
    instruction ='''Refine the following transcript by performing the following tasks:

    1. Remove filler words and phrases such as "um", "uh", "you know", "like", "so", "okay", "right", "I mean".
    2. Correct any grammatical errors, including subject-verb agreement, tense consistency, punctuation, and sentence fragments.
    3. Eliminate repetitive or irrelevant content while maintaining the main points and context of the discussion.
    4. Improve sentence structure and coherence to ensure clarity and readability.
    5. Organize the text into paragraphs or sections based on the topics or themes discussed.

    Ensure that the final text is clear, coherent, and useful for generating questions.

    Example:

    Raw Transcript:
    "So, um, today we're going to talk about machine learning, uh, you know, it's a really interesting topic. Um, like, there are different algorithms, you know, like neural networks and, uh, decision trees."

    Refined Text:
    "There are various algorithms such as neural networks and decision trees."
    '''
    context = llm.invoke(instruction+transcript,temperature=1, max_new_tokens=2000)
    return context


def CompareAnswersInovke(question:str,correct_answer:str,student_answer:str,  llm):
    instruction= f"""
                    
                    Please provide the following:
                    1. Speek as teacher
                    2. Feedback on the student's answer.
                    3. A brief explanation for the score.

                    Example:
                    Question: What is the capital of France?
                    Correct Answer: The capital of France is Paris. It is known for its culture and history.
                    Student Answer: Paris is the capital city of France, famous for its historical landmarks and rich culture.

                    Feedback: The student's answer correctly identifies Paris as the capital of France and highlights its cultural and historical significance. The phrasing differs slightly, but the content and meaning are almost identical.
                    Explanation: Both answers identify Paris as the capital of France and highlight its cultural and historical significance. The slight difference in phrasing does not affect the overall accuracy and completeness of the response.

                    Question: {question}
                    Correct Answer: {correct_answer}
                    Student Answer: {student_answer}

                    """

    context = llm.invoke(instruction,temperature=1, max_new_tokens=2000)
    return {"feedback":context}    


def adjust_prob_distribution(types_to_include:list):
    '''
    This probability distribution of question is matching the our collected data
    '''
    prob_distribution = [0.6 , 0.2 , 0.2]  # probabilities: [MCQ, True/False, Open]

    if len(types_to_include) == 1:
        # If only one type is to be included, set its probability to 1
        prob_distribution = [0 , 0 , 0]
        prob_distribution[types_to_include[0]] = 1.0
    elif len(types_to_include)  == 2:
        # Adjust the probability distribution proportionally
        prob_distribution = [0 , 0 , 0]
        types_to_include = sorted(types_to_include)
        prob_distribution[types_to_include[0]] = 0.6
        prob_distribution[types_to_include[1]] = 0.4
    
    return prob_distribution


def load_generation_generation_model():

    model = AutoPeftModelForCausalLM.from_pretrained(
    "shredder-31/Llamma-3_QG_V.2.0",
    load_in_4bit = True,
    low_cpu_mem_usage = True
    ).to('cuda')

    tokenizer = AutoTokenizer.from_pretrained("shredder-31/Llamma-3_QG_V.2.0")
    return model , tokenizer


def split_transcrpit(transcrpit:str,prob_distribution:list,chunk_size:int=2000, chunk_overlap:int=1000):
    '''
    My question generation model generate one question at time and we need more question in same context so by spilting 
    the main points using scpey sentencizer tokenizer we can chunk them in groups this way can question more then question 
    '''
    new_chunks = []
    chunks = text_splitter(transcrpit,type=3,chunk_size=chunk_size, chunk_overlap=chunk_overlap)
 
    for chunk in chunks:

        t = np.random.choice([0,1,2], p=prob_distribution)

        instruction = ""

        if t == 0:
            instruction = "You will assist me in generating MCQ questions along with their Answers and Choices. Please use the next context to guide you generating of MCQ questions### Context: : "

        elif t == 1:
            instruction = "You will assist me in generating True or False questions along with their Answers. Please use the next context to guide you generating of True or False questions### Context: : "     

        else :
            instruction = "You will assist me in generating Open questions along with their Answers. Please use the next context to guide you generating of Open questions### Context: : "

        chunk += instruction
        
        new_chunks.append(chunk)

    return new_chunks


def Generate_Questions_inferenace(model, tokenizer, chunk_texts):
    results = {}
    i = 0
    model.eval()  
    with torch.no_grad():
        for chunk in chunk_texts:
            inputs = tokenizer(
                [
                    alpaca_prompt.format(
                        f"{chunk}",
                        "",
                    )
                ], return_tensors="pt").to("cuda")

            outputs = model.generate(**inputs,temperature=0.9,top_p=0.85, max_new_tokens=100, use_cache=True)
            outputs = tokenizer.batch_decode(outputs)
           
            outputs = parse_text(outputs[0])
            results[i] = outputs
            i+=1
            # Clean up memory
            del inputs
            torch.cuda.empty_cache()
    
    return results

def question_generation_query(model,llm, tokenizer, transcript:str, types:list, chunk_size:int, chunk_overlap:int):
    context = clean_transcript(transcript,llm)
    
    prob_distribution = adjust_prob_distribution(types)

    print("Prob"," ", prob_distribution)

    chunk_texts = split_transcrpit(context, prob_distribution, chunk_size, chunk_overlap)  

    output = Generate_Questions_inferenace(model, tokenizer, chunk_texts)
  
    return  output

    

