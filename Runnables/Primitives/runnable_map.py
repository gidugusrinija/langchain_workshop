from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence, RunnableMap
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

# template 1 : tweet generation

template1 = PromptTemplate(
    template="Generate a tweet about: {topic}",
    input_variables=["topic"]
)

# template 2: linkedin post generation
template2 = PromptTemplate(
    template="Generate a linkedin post about: {topic}",
    input_variables=["topic"])

parser = StrOutputParser()

model1 = ChatOpenAI(model='gpt-4o', temperature=0.5)
model2 = ChatOpenAI(model='gpt-4o-mini', temperature=0.5)

parallel_chain = RunnableMap({
    "tweet": RunnableSequence(template1, model1, parser),
    "linkedin": RunnableSequence(template2, model1, parser)
})
res = parallel_chain.invoke({"topic": "Artificial Intelligence"})
print(res)

