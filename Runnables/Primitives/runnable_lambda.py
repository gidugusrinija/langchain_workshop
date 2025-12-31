from langchain_openai import ChatOpenAI
from langchain_core.runnables import (RunnableSequence, RunnableParallel,
                                      RunnablePassthrough, RunnableLambda)
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


def word_counter(line: str) -> int:
    return len(line.split())


runnable_word_counter = RunnableLambda(word_counter)

model = ChatOpenAI(model='gpt-4o-mini')

template1 = PromptTemplate(
    template="Tell a joke about: {topic}",
    input_variables=["topic"])

template2 = PromptTemplate(
    template="Merge the following joke and the word count of the joke into a single string:\nJoke: {joke}\nWord Count: {count}\n Give only asked output format dont return schema",
    input_variables=["joke", "count"])

parser = StrOutputParser()


chain1 = RunnableSequence(template1, model, parser)
chain2 = RunnableParallel({
    "joke": RunnablePassthrough(),
    "count": runnable_word_counter,
})
final_chain = RunnableSequence(
    chain1,
    chain2,
    RunnableSequence(template2, model, parser)
)
res = final_chain.invoke({"topic": "India"})

print(res)
