from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)
documents = ["New Delhi is the capital of India",
             "Paris is the capital of France",
             "Canberra is the capital of Australia"]
result = embedding.embed_query("New Delhi is the capital of India")
result_2 = embedding.embed_documents(documents)
print(str(result_2), type(result_2))
print(str(result), type(result))
