"""
In latest langchain version, StructuredOutputParser is no longer supported.
"""
import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema

from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
    task="text-generation"
)
chat_model = ChatHuggingFace(llm=llm)

# Schema
schema = [
    ResponseSchema(name="name", description="The name of the fictional person"),
    ResponseSchema(name="age", description="The age of the fictional person"),
    ResponseSchema(name="city", description="The city where the fictional person lives"),
]

# Create Structured Output Parser
parser = StructuredOutputParser.from_response_schemas(schema)

# Template using JSON Output Parser

template = PromptTemplate(template="Give me 3 personal features of a fictional person.{format_instructions}",
                          input_variables=[],
                          partial_variables={"format_instructions": parser.get_format_instructions()})

chain = template | chat_model | parser

final_result = chain.invoke({})

print(final_result)



