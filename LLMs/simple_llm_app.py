from langchain_openai import OpenAI
# from langchain_openai.llms import OpenAI
# both imports refers to the same OpenAI LLM class
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()
llm = OpenAI(model='gpt-3.5-turbo-instruct')
user_query = input("Enter your question: ")
template = PromptTemplate(
    template="Answer the following question {question}",
    input_variables=["question"]
)
prompt = template.format(question=user_query)

result = llm.invoke(prompt)
# result2 = llm.predict(prompt)
# LangChain moved everything to the Runnable API, where the standard method is: invoke()
# In modern LangChain, predict() is gone — always use invoke() instead.
print(result)
