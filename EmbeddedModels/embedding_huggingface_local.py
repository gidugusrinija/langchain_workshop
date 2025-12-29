from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
text = "Delhi is the capital of India"
documents = [
    "New Delhi is the capital of India",
    "Paris is the capital of France",
    "Canberra is the capital of Australia"
]

vector = embeddings.embed_query(text)
vector2 = embeddings.embed_documents(documents)

print(str(vector), type(vector))
print(str(vector2), type(vector2))
