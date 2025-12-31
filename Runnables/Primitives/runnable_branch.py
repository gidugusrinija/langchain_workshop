from langchain_openai import ChatOpenAI
from langchain_core.runnables import (RunnableSequence, RunnableLambda,
                                      RunnableBranch, RunnablePassthrough)
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


def word_counter(line: str) -> dict:
    return {"count": len(line.split()), "report": line}


runnable_word_counter = RunnableLambda(word_counter)

model = ChatOpenAI(model='gpt-4o-mini')

template1 = PromptTemplate(
    template="Give me a detailed report on this topic: {topic}",
    input_variables=["topic"])

template2 = PromptTemplate(
    template="Summarize the following report which should be less than or equal to 500 words:\nReport: {report}",
    input_variables=["report"])

parser = StrOutputParser()


chain1 = RunnableSequence(template1, model, parser)

# chain2 = RunnableBranch((lambda x: x["count"] >= 500, RunnableSequence(template2, model, parser)),
#                         (lambda x: x["count"] < 500, RunnablePassthrough()),
#                         RunnableLambda(lambda x: "Not a valid word count"))
# If report count < 500, output will be dict because of passthrough

chain2 = RunnableBranch((lambda x: x["count"] >= 500, RunnableSequence(template2, model, parser)),
                        (lambda x: x["count"] < 500, RunnableLambda(lambda x: x["report"])),
                        RunnableLambda(lambda x: "Not a valid word count"))
final_chain = RunnableSequence(
    chain1,
    runnable_word_counter,
    chain2
)
res = final_chain.invoke({"topic": "India"})

print(res)
