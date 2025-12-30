from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Define a simple prompt template
template = PromptTemplate(
    template="Tell me 5 interesting facts about {topic}.",
    input_variables=["topic"]
)

# Define simple parser
parser = StrOutputParser()

# Create Model
model = ChatOpenAI(model='gpt-4o', temperature=0)

# Create Simple chain
chain = template | model | parser

result = chain.invoke({"topic": "Lord Sriram"})

print(result)

chain.get_graph().print_ascii()
