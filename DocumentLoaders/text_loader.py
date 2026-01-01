from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
loader = TextLoader("cricket.txt", encoding="utf-8")

parser = StrOutputParser()

model = ChatOpenAI(model='gpt-4o-mini')

docs = loader.load()

template = PromptTemplate(template="Generate a 3 line summary for this report: \n {text}",
                          input_variables=["text"])

summary_chain = template | model | parser

res = summary_chain.invoke({"text": docs[0].page_content})

print(res)
