from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


# Source documents
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models.")
]

# Initialize the embedding model
embedding_model = OpenAIEmbeddings()

# Create chroma vector store in memory
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="my_sample_collection"
)

# Convert Vector store into a retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

query = "What is Chroma used for?"
result = retriever.invoke(query)

for idx, val in enumerate(result):
    print(f"____Result:{idx + 1}_____")
    print(f"Content: \n{val.page_content}\n")
