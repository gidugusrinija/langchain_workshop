from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
loader = PyPDFLoader("Pothole_2023.pdf")  # Mention complete path if it is not in the same directory

docs = loader.load()

print(docs[0].page_content)
