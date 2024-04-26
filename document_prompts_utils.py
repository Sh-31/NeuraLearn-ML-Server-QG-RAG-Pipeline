from langchain_core.prompts import PromptTemplate # type: ignore
from langchain_text_splitters import TokenTextSplitter , RecursiveCharacterTextSplitter # type: ignore
from langchain_text_splitters.spacy import SpacyTextSplitter   # type: ignore
from langchain_community.embeddings import HuggingFaceEmbeddings # type: ignore



template_string = """<bos><start_of_turn>user
DOCUMENT: {context} 
QUESTION: {query} 
INSTRUCTIONS:
Answer the users QUESTION using the DOCUMENT text above.
Keep your answer ground in the facts of the DOCUMENT.
If the DOCUMENT doesn’t contain the facts to answer the QUESTION return NONE
<end_of_turn>
<start_of_turn>model
"""


prompt = PromptTemplate(
    input_variables=["context","query"],
    template=template_string,
)

def text_splitter(text:str, type:int = 1, chunk_size:int=2000, chunk_overlap:int=150 ,separators:list[str] =["\n\n", "\n", " ", ""])->list[str]:
    """
    Splits the input text into chunks based on the specified type of text splitter.

    Parameters:
        text (str): The input text to be split.
        type (int, optional): The type of text splitter to use. 
            1 for RecursiveCharacterTextSplitter (default), 
            2 for TokenTextSplitter (More Efficient for a longer text), 
            any other value for SpacyTextSplitter (more Smarter).
        chunk_size (int, optional): The maximum size of each chunk (default is 4000).
        chunk_overlap (int, optional): The overlap between consecutive chunks (default is 200).
        separators (list[str], optional): List of separator strings used to split the text (default is ["\n\n", "\n", " ", ""]).
    
    Returns:
        list[str]: A list of text chunks.
    """
    splitter = None
    if type == 1:
         splitter = RecursiveCharacterTextSplitter(
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap, 
                                separators=separators
                            )
    elif type == 2:
         splitter = TokenTextSplitter()

    else:
        
        splitter = SpacyTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator = "\n",  pipeline = 'sentencizer') 
         
    
    return splitter.split_text(text)


def embeddings_texts(splites:list[str])->list[int]:
     model_name = "sentence-transformers/all-mpnet-base-v2"
     model_kwargs = {'device': 'cpu'}

     encode_kwargs = {'normalize_embeddings': False}

     hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
     )
     embeddings_texts = []   
     for query in splites:
         embeddings_texts.append(hf.embed_querylitt(query))

     return embeddings_texts


def embedding_model(runtime="Cuda"):
     model_name = "sentence-transformers/all-mpnet-base-v2"
     model_kwargs = {'device': runtime}

     encode_kwargs = {'normalize_embeddings': False}
     hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
     )
     return hf
     