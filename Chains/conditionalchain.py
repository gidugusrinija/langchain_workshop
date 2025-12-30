from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatOpenAI()


class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="The sentiment of the feedback text")


pydanticparser = PydanticOutputParser(pydantic_object=Feedback)

parser = StrOutputParser()

template = PromptTemplate(
    template="Classify the following feedback text into Positive or Negative: \n {text} \n {format_instruction}",
    input_variables=["text"],
    partial_variables={"format_instruction":pydanticparser.get_format_instructions()})

chain1 = template | model | pydanticparser



positive_template = PromptTemplate(
    template="Write appropriate response for the following positive feedback:\n {text}",
    input_variables=["text"]
)

negative_template = PromptTemplate(
    template="Write appropriate response for the following negative feedback:\n {text}",
    input_variables=["text"]
)

positive_chain = positive_template | model | parser
negative_chain = negative_template | model | parser

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", positive_chain),
    (lambda x: x.sentiment == "negative", negative_chain),
    RunnableLambda(lambda x: "Could not determine sentiment.")
)

final_chain = chain1 | branch_chain

result = final_chain.invoke({"text": "It is such a terrible experience. I didnt even want to give 0 star rating"})
print(result)