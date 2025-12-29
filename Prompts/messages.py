from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI(model='gpt-4o', temperature=0)

messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="Hello, Tell me about langchain")]
response = model.invoke(messages)
print(response.content)
messages.append(AIMessage(content=response.content))
print(messages)
