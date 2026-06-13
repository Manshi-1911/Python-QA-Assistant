import pandas as pd

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

df = pd.read_csv("data/python_questions.csv")

documents = []

for _, row in df.iterrows():

    text = f"""
    Question:
    {row['Title']}

    Answer:
    {row['Body']}
    """

    documents.append(text)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_texts(
    documents,
    embeddings
)

db.save_local("vectorstore")

print("Vector store created")
