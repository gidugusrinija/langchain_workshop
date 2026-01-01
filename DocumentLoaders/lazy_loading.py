from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

loader = DirectoryLoader(
    path="pdffolder",
    glob="*.pdf",
    loader_cls=PyPDFLoader)

docs = loader.lazy_load()  # Generator
# print(len(doc))
# len(docs) may not reflect the actual number of documents until fully loaded. May throw error
print("Document contents preview:", docs)
parser = StrOutputParser()
model = ChatOpenAI(model='gpt-4o-mini')

for doc in docs:
    print(doc)

