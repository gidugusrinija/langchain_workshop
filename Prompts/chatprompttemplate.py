from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# Ambiguous way
chat_template = ChatPromptTemplate(
    [
        SystemMessage(content="You are a helpful {domain} expert."),
        HumanMessage(content="Explain in simple terms, what is {topic}"),
    ])
prompt = chat_template.invoke({
    "domain": "cricket",
    "topic": "Dusra"})
print(prompt, end="\n\n")


# Approach 1: Using tuples
chat_template1 = ChatPromptTemplate(
    [
        ("system", "You are a helpful {domain} expert."),
        ("human", "Explain in simple terms, what is {topic}")
    ]
)
prompt2 = chat_template1.invoke({
    "domain": "cricket",
    "topic": "Dusra"
})
print(prompt2, end="\n\n")


# Approach 2: Using from_messages method

chat_template3 = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful {domain} expert."),
        ("human", "Explain in simple terms, what is {topic}")
    ]
)

prompt3 = chat_template3.invoke({
    "domain": "cricket",
    "topic": "Dusra"
})
print(prompt3)
