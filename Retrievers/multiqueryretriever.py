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
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=ChatOpenAI(model="gpt-3.5-turbo")
)
# MultiQueryRetriever is just a wrapper not retriever itself where it will take a retriever and llm as input
# Where it uses llm to get different variations for same user query and pass all variations to retriever
# and merges all output. here you may see the no.of outputs more than search kwargs
# because it's the kwargs for each retrieval not after combination
# Uses LLM before retrieval

# Query
query3 = "How to improve energy levels and maintain balance?"

similarity_results = similarity_retriever.invoke(query3)
multiquery_results = multiquery_retriever.invoke(query3)

for i, doc in enumerate(similarity_results):
    print(f"____Result:{i + 1}_____")
    print(f"Content: \n{doc.page_content}\n")

print("*"*150)

for i, doc in enumerate(multiquery_results):
    print(f"____Result:{i + 1}_____")
    print(f"Content: \n{doc.page_content}\n")
