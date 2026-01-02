from langchain_community.retrievers import WikipediaRetriever

# Initialize the retriever(optional: set language and top_k)
retriever = WikipediaRetriever(top_k_results=2, lang="en")

# Define query
query = "The geopolitical history of India and Pakistan from the perspective of a chinese"

# Get wikipedia documents
docs = retriever.invoke(query)

for idx, val in enumerate(docs):
  print(f"____Result:{idx + 1}_____")
  print(f"Content: \n{val.page_content}\n")

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
retriever = vector_store.as_retriever(search_kwargs={"k":2})

query = "What is Chroma used for?"
result = retriever.invoke(query)

for idx, val in enumerate(result):
  print(f"____Result:{idx + 1}_____")
  print(f"Content: \n{val.page_content}\n")

# MMR: Maximum Marginal Relevance
# Sample documents
docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more.")
]

from langchain_community.vectorstores import FAISS

embedding2 = OpenAIEmbeddings()

# create faiss vector store
vector_s = FAISS.from_documents(
    documents=docs,
    embedding=embedding2
)

faiss_retriever = vector_s.as_retriever(
    search_type="mmr",
    search_kwargs={"k":3, "lambda_mult":1}
)

query2 = "What is Langchain?"
results = faiss_retriever.invoke(query2)

for idx, val in enumerate(results):
  print(f"____Result:{idx + 1}_____")
  print(f"Content: \n{val.page_content}\n")


"""# Multiquery Retrieval"""

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_classic.retrievers import MultiQueryRetriever

# Relevant health & wellness documents
all_docs = [
    Document(
        page_content="Regular walking boosts heart health and can reduce symptoms of depression.",
        metadata={"source": "H1"}
    ),
    Document(
        page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.",
        metadata={"source": "H2"}
    ),
    Document(
        page_content="Deep sleep is crucial for cellular repair and emotional regulation.",
        metadata={"source": "H3"}
    ),
    Document(
        page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.",
        metadata={"source": "H4"}
    ),
    Document(
        page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.",
        metadata={"source": "H5"}
    ),
    Document(
        page_content="The solar energy system in modern homes helps balance electricity demand.",
        metadata={"source": "I1"}
    ),
    Document(
        page_content="Python balances readability with power, making it a popular system design language.",
        metadata={"source": "I2"}
    ),
    Document(
        page_content="Photosynthesis enables plants to produce energy by converting sunlight.",
        metadata={"source": "I3"}
    ),
    Document(
        page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.",
        metadata={"source": "I4"}
    ),
    Document(
        page_content="Black holes bend spacetime and store immense gravitational energy.",
        metadata={"source": "I5"}
    )
]

# Initialize the embedding
embedding3 = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents=all_docs,
                                   embedding=embedding3,
                                   )

similarity_retriever = vectorstore.as_retriever(search_type="similarity",
                                                search_kwargs={"k": 5})

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k":5}),
    llm=ChatOpenAI(model="gpt-3.5-turbo")
)

# Query
query3 = "How to improve energy levels and maintain balance?"

similarity_results = similarity_retriever.invoke(query3)
multiquery_results = multiquery_retriever.invoke(query3)

for i,doc in enumerate(similarity_results):
  print(f"____Result:{i + 1}_____")
  print(f"Content: \n{doc.page_content}\n")

print("*"*150)

for i,doc in enumerate(multiquery_results):
  print(f"____Result:{i + 1}_____")
  print(f"Content: \n{doc.page_content}\n")

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

docs = [
    Document(
        page_content="""
The Grand Canyon is one of the most visited natural wonders in the world.
Photosynthesis is the process by which green plants convert sunlight into energy.
Millions of tourists travel to see it every year. The rocks date back millions of years.""",
        metadata={"source": "Doc1"}
    ),

    Document(
        page_content="""
In medieval Europe, castles were built primarily for defense.
Chlorophyll in plant cells captures sunlight during photosynthesis.
Knights wore armor made of metal. Siege weapons were often used to breach castle walls.""",
        metadata={"source": "Doc2"}
    ),

    Document(
        page_content="""
Basketball was invented by Dr. James Naismith in the late 19th century.
It was originally played with a soccer ball and peach baskets.
NBA is now a global league.""",
        metadata={"source": "Doc3"}
    ),

    Document(
        page_content="""The history of cinema began in the late 1800s.
Silent films were the earliest form.Thomas Edison was among the pioneers.
Photosynthesis does not occur in animal cells. Modern film making involves complex CGI and sound design.""",
        metadata={"source": "Doc4"}
    ),
]

embedding4 = OpenAIEmbeddings()
vecstor = FAISS.from_documents(documents=docs, embedding=embedding4)

llm = ChatOpenAI(model="gpt-3.5-turbo")
compressor = LLMChainExtractor.from_llm(llm)

base_retriever = vecstor.as_retriever(search_kwargs={"k":5})

compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor
)

query = "what is photosynthesis?"
compressed_res = compression_retriever.invoke(query)

for i, doc in enumerate(compressed_res):
  print(f"____Result:{i + 1}_____")
  print(f"Content: \n{doc.page_content}\n")

