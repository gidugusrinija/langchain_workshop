from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more.")
]


embedding2 = OpenAIEmbeddings()

# create faiss vector store
vector_s = FAISS.from_documents(
    documents=docs,
    embedding=embedding2
)

faiss_retriever = vector_s.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "lambda_mult": 1}  # less the value lambda_mult, more the diversity
)

query2 = "What is Langchain?"
results = faiss_retriever.invoke(query2)

for idx, val in enumerate(results):
  print(f"____Result:{idx + 1}_____")
  print(f"Content: \n{val.page_content}\n")