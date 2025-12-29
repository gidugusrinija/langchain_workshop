from langchain_core.prompts import (PromptTemplate, ChatPromptTemplate,
                                    FewShotPromptTemplate)

# Dynamic and re-usable prompts
prompt = PromptTemplate.from_template("Summarize {topic} in {emotion} tone")
print(prompt.format(topic="Cricket", emotion="happy"))  # Summarize Cricket in happy tone

# Role based prompts
chat_prompt = ChatPromptTemplate.from_messages([("system", "Hi, you are a experienced {profession}"),
                                                ("user", "Can you tell me about {topic}?"),
                                                ])
print(chat_prompt.format(profession="doctor", topic="health"))
print(chat_prompt.format_messages(profession="doctor", topic="health"))


# Few Shot Prompts
examples = [
    {"input": "What is LangChain?",
     "output": "LangChain is a framework for developing applications powered by language models."},
    {"input": "Explain Prompt Engineering.",
     "output": "Prompt Engineering is the process of designing and refining prompts to effectively interact with language models."},
    {"input": "Define ChatGPT.",
     "output": "ChatGPT is a conversational AI model developed by OpenAI that can engage in human-like dialogue."},
    {"input": "What is few-shot learning?",
     "output": "Few-shot learning is a machine learning approach where a model learns to perform tasks with only a small number of training examples."}
]

example_template = """
Question: {input}
Answer: {output}
"""
example_prompt = PromptTemplate(input_variables=["input", "output"],
                                template="Question: {input}\nAnswer: {output}")
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Here are some examples of questions and answers:",
    suffix="Now, answer the following question:\nQuestion: {question}\nAnswer:",
    input_variables=["question"],
)
print(few_shot_prompt.format(question="What is LangChain?"))
print(few_shot_prompt.format(question="What is Python?"))
