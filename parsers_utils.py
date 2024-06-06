import re

def parse_text(text):

    question_pattern = r'### Question:\s*\n:\s*(.*?)\s*\n'
    answer_pattern = r'### Answer:\s*\n:\s*(.*?)(?=\s*<\|end_of_text\|>|\n|$)'  
    choices_pattern = r'### Choices:\s*\n(.*?)\s*(?=\n###|\Z)' 


    question_match = re.search(question_pattern, text, re.DOTALL)
    question = question_match.group(1).strip() if question_match else None

    
    answer_match = re.search(answer_pattern, text, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else None


    choices_match = re.search(choices_pattern, text, re.DOTALL)
    choices = [choice.strip() for choice in choices_match.group(1).strip().split('\n')] if choices_match else None

    return {
        'Question': question,
        'Answer': answer,
        'Choices': choices
    }