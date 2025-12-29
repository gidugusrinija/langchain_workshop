from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
documents = ["New Delhi is the capital of India",
             "Paris is the capital of France",
             "Canberra is the capital of Australia",
             "Tokyo is the capital of Japan",
             "Apple is a technology company",
             "Microsoft develops software products",
             "The Eiffel Tower is in Paris",
             "Virat Kohli is a famous cricket player who plays for India"]


query = "Tell me about virat kohli"

document_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)
scores = cosine_similarity([query_embedding], document_embeddings)[0]
index, score = sorted(enumerate(scores), key=lambda x: x[1])[-1]
print(documents[index])
print(f"Similarity Score: {score}")
