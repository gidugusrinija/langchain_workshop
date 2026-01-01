from langchain_community.document_loaders import CSVLoader
from dotenv import load_dotenv

load_dotenv()
loader = CSVLoader(file_path="train_v2_train_v2_heart_classification.csv")  # Mention complete path if it is not in the same directory

docs = loader.lazy_load()

for doc in docs:
    print(doc.page_content)
