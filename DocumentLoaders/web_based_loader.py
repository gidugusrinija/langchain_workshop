from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

url = "https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421"
loader = WebBaseLoader(url)

docs = loader.load()

parser = StrOutputParser()

model = ChatOpenAI(model='gpt-4o-mini')


template = PromptTemplate(template="Answer the following question: \n {question} from the following text: \n {text}",
                          input_variables=["question", "text"])


summary_chain = template | model | parser

res = summary_chain.invoke({"text": docs[0].page_content, "question": "What is the price of the MacBook Air M2?"})
print(res)
