from DummyLLM import DummyLLM, DummyPromptTemplate, DummyStrOutputParser
from DummyRunnable import RunnableConnector

llm = DummyLLM()
template1 = DummyPromptTemplate(
    template="Provide a detailed report on the following topic:{topic}",
    input_variables=["topic"]
)
parser = DummyStrOutputParser()

chain = RunnableConnector([template1, llm, parser])

result = chain.invoke({"topic": "Tell me one random fact"})

print(result)