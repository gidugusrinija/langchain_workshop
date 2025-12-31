from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(model='gpt-4o-mini')

template1 = PromptTemplate(
    template="Tell a joke about: {topic}",
    input_variables=["topic"])

template2 = PromptTemplate(
    template="Explain the joke: {joke}",
    input_variables=["joke"])

parser = StrOutputParser()

chain1 = RunnableSequence(template1, model, parser)
chain2 = RunnableParallel({"joke": RunnablePassthrough(), "explanation": RunnableSequence(template2, model, parser)})
final_chain = RunnableSequence(chain1, chain2)
res = final_chain.invoke({"topic": "chickens"})
print(res)

