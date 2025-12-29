from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()
# chat template
chat_template = ChatPromptTemplate(
    [
        ("system", "You are a helpful customer support agent."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", '{query}')],
    input_variables=['query']
)

chat_history = []

# load chat history
with open('chat_history.txt', 'r') as file:
    chat_history.extend(file.readlines())

# print(chat_history)

while True:
    user_query = input("You: ")

    if user_query.lower() in ['exit', 'quit']:
        print("AI: Exiting the chatbot. Have a Nice day!")
        break

    final_prompt = chat_template.invoke({
        "chat_history": chat_history,
        "query": user_query
    })
    model = ChatOpenAI(model='gpt-4o', temperature=0.5)
    response = model.invoke(final_prompt)

    with open('chat_history.txt', 'a') as file:
        file.write("HumanMessage(content=" + user_query + ")\n")
        file.write("AIMessage(content=" + response.content + ")\n")
        print("AI: ", response.content)
