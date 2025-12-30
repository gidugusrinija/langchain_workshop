from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel

load_dotenv()


chat_model1 = ChatOpenAI(model='gpt-4o-mini', temperature=0)  # used in parallel chains
chat_model2 = ChatOpenAI(model='gpt-4o', temperature=0)  # used in merge

# 1st prompt: Detailed Report of the topic
template1 = PromptTemplate(
    template="Generate the short and simple notes from the following text:\n{text}",
    input_variables=["text"]
)

# 2nd Prompt: 5 lines summary of the topic
template2 = PromptTemplate(
    template="Generate 5 short question answers from the following text:\n{text}",
    input_variables=["text"]
)

template3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}",
    input_variables=["notes", "quiz"]
)

# Using StrOutputParser

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "notes": template1 | chat_model1 | parser,
    "quiz": template2 | chat_model1 | parser
})

merge_chain = parallel_chain | template3 | chat_model2 | parser

text = """
The old watchmaker opened his shop every morning at the same time, winding the clocks before the city fully woke up. Each tick echoed softly through the room, steady and patient, like a promise that time could still be trusted. People said his watches never ran late because he treated time with respect.

One rainy afternoon, a child wandered in holding a broken watch that no longer ticked. The watchmaker didn’t ask for money; he only asked the child to sit and listen. As he repaired the gears, he explained how even the smallest piece mattered, and how patience could fix things that force never could.

When the watch finally started ticking again, the rain outside had stopped. The child left smiling, and the watchmaker returned to his work, comforted by the sound of time moving forward—quietly, reliably, and always with purpose.

"""

result = merge_chain.invoke({"text": text})

print(result)

merge_chain.get_graph().print_ascii()
