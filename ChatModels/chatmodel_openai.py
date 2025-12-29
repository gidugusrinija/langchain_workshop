from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
chat_model = ChatOpenAI(model='gpt-4', temperature=2, max_completion_tokens=30)
result = chat_model.invoke("What is the capital of India?")
print(result.content)
