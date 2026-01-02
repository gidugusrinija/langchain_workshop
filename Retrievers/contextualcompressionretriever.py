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

"""
ContextualCompressionRetriever is also a wrapper, not a retriever by itself.
It takes a base retriever and an LLM-based compressor as input.

The base retriever is responsible for fetching the top-k documents
using vector similarity or any other retrieval strategy.

After retrieval, the LLM is used to compress each document by
extracting only the parts that are relevant to the user query.
Irrelevant sentences or sections are removed.

The final output contains the same number of documents,
but with reduced and more focused content.

ContextualCompressionRetriever uses the LLM after retrieval
to improve precision and reduce noise in the context.
"""

query = "what is photosynthesis?"
compressed_res = compression_retriever.invoke(query)

for i, doc in enumerate(compressed_res):
    print(f"____Result:{i + 1}_____")
    print(f"Content: \n{doc.page_content}\n")
