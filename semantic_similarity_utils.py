from langchain_community.vectorstores import Chroma# type: ignore

def vector_db(text,embdedding):
    db = Chroma.from_texts(text, embedding=embdedding)
    return db


def retrieval(vector_db,query,k=3):
    return vector_db.similarity_search_with_relevance_scores(query, k=k)
