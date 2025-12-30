import os

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
    task="text-generation"
)
chat_model = ChatHuggingFace(llm=llm)

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

prompt1 = template1.invoke({"topic": "Climate Change"})

result = chat_model.invoke(prompt1)

print(result.content)

prompt2 = template2.invoke({"text": result.content})

result2 = chat_model.invoke(prompt2)

print("\n\n******************\n\n5 line summary: ", result2.content, "\n\n******************\n\n")


# Using StrOutputParser

parser = StrOutputParser()

chain = template1 | chat_model | parser | template2 | chat_model | parser

final_result = chain.invoke({"topic": "Climate Change"})

print("\n\n******************\n\n5 line summary using StrOutputParser: ", final_result, "\n\n******************\n\n")
