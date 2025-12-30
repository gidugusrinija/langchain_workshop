from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatOpenAI(model='gpt-4o', temperature=0)

# 1st prompt: Detailed Report of the topic
template1 = PromptTemplate(
    template="Provide a detailed report on the following topic:{topic}",
    input_variables=["topic"]
)

# 2nd Prompt: 5 lines summary of the topic
template2 = PromptTemplate(
    template="Write a 5 line summary on the following text:\n{text}",
    input_variables=["text"]
)

# Using StrOutputParser

parser = StrOutputParser()

chain = template1 | chat_model | parser | template2 | chat_model | parser

final_result = chain.invoke({"topic": "Climate Change"})

print(final_result)
