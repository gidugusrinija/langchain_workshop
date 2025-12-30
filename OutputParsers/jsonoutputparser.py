import os

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
    task="text-generation"
)
chat_model = ChatHuggingFace(llm=llm)

# Create JsonOutputParser
parser = JsonOutputParser()

# Template using JSON Output Parser

template = PromptTemplate(template="Give me name, age, city of a fictional person.{format_instructions}",
                          input_variables=[],
                          partial_variables={"format_instructions": parser.get_format_instructions()})
prompt = template.invoke({})
prompt2 = template.format()  # Not Runnable in chains
print(prompt)
print(prompt2)

chain = template | chat_model | parser

final_result = chain.invoke({})

print(final_result)

