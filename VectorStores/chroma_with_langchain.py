from langchain_openai import OpenAIEmbeddings

import os

from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma

from langchain_core.documents import Document

# Create LangChain documents for IPL players

load_dotenv()

doc1 = Document(
    page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and strong leadership.",
    metadata={"team": "Royal Challengers Bangalore"}
)

doc2 = Document(
    page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He is known for his calm leadership and powerful batting.",
    metadata={"team": "Mumbai Indians"}
)

doc3 = Document(
    page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills and tactical mind are legendary.",
    metadata={"team": "Chennai Super Kings"}
)

doc4 = Document(
    page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his deadly yorkers.",
    metadata={"team": "Mumbai Indians"}
)

doc5 = Document(
    page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, he is also an excellent fielder.",
    metadata={"team": "Chennai Super Kings"}
)

documents = [doc1, doc2, doc3, doc4, doc5]

vector_store = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory="my_chroma_db",
    collection_name="sample"
)

vector_store.add_documents(documents)

# view data
res1 = vector_store.get(include=["embeddings", "documents", "metadatas"])

# search documents
res2 = vector_store.similarity_search(query="Who among these are bowlers", k=2)

res3 = vector_store.similarity_search_with_score(query="who among these are bowlers", k=2)

# metadata filtering
res4 = vector_store.similarity_search_with_score(query="who among these are bowlers", k=3,filter={"team": "Mumbai Indians"})

# update documents
updated_doc1 = Document(
    page_content="Virat Kohli is my favourite cricket player. He is former captain. His jersey number is 18",
    metadata={"team": "Royal Challengers Bangalore"}
)
vector_store.update_document(document_id="32df581e-2cf5-4e02-a8ab-5da997bb9be7", document=updated_doc1)

# view updated documents
res5 = vector_store.get(include=["documents"])

# delete document
vector_store.delete(ids=["b2b33bb3-4807-4d67-8078-d8b5278d512c"])

# view after deleting
res6 = vector_store.get(include=["documents"])

print(res1, res2, res3, res4, res5, res6, sep="\n\n")

