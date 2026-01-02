from langchain_community.retrievers import WikipediaRetriever

# Initialize the retriever(optional: set language and top_k)
retriever = WikipediaRetriever(top_k_results=2, lang="en")

# Define query
query = "The geopolitical history of India and Pakistan from the perspective of a chinese"

# Get wikipedia documents
docs = retriever.invoke(query)  # Runnable

for idx, val in enumerate(docs):
    print(f"____Result:{idx + 1}_____")
    print(f"Content: \n{val.page_content}\n")
