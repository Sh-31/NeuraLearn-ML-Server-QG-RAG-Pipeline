from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain , create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain



#  create_history_aware_retriever constructs a chain that accepts keys input and chat_history as input, and has the same output schema as a retriever.
# https://python.langchain.com/docs/use_cases/question_answering/chat_history/


def chat(llm,question,retriever,chat_history=[],add_to_history=True):
    

    qa_system_prompt ="""You are an NeuraLearnAcademy Ai assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    history_aware_retriever = create_history_aware_retriever(
        llm,  retriever, qa_prompt
    )


    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    result = rag_chain.invoke({"input":question, "chat_history": chat_history,"context":""})
    if add_to_history: 
        chat_history.extend([question, result["answer"]])
    return result , chat_history