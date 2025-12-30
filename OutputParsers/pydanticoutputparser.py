import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser

from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
    task="text-generation"
)
chat_model = ChatHuggingFace(llm=llm)


# Schema
class Person(BaseModel):
    name: str = Field(description="The name of the fictional person")
    age: int = Field(description="The age of the fictional person")
    city: str = Field(description="The city where the fictional person lives")


# Create Structured Output Parser
parser = PydanticOutputParser(pydantic_object=Person)

# Template using Pydantic Output Parser

template = PromptTemplate(
    template="""
Generate a fictional person from {country}.

IMPORTANT:
- Return ONLY a JSON object
- Fill in REALISTIC VALUES for all fields
- Do NOT return the schema
- Do NOT explain anything
- Do NOT use markdown

{format_instructions}
""",
    input_variables=["country"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

chain = template | chat_model | parser

final_result = chain.invoke({"country": "Indian"})

print(final_result.model_dump())



