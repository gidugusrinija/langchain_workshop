from DummyRunnable import Runnable
import random


class DummyLLM(Runnable):
    def __init__(self):
        print("LLM Created")

    def invoke(self, prompt: str) -> dict:
        responses = ["Andhra Pradesh is the rice bowl of India",
                     "The capital of India is New Delhi",
                     "The largest mammal is the blue whale",
                     "The tallest mountain is Mount Everest"]
        return {"content": random.choice(responses)}


class DummyPromptTemplate(Runnable):
    def __init__(self, template: str, input_variables: list):
        self.template = template
        self.input_variables = input_variables

    def invoke(self, input_data) -> str:
        return self.template.format(**input_data)


class DummyStrOutputParser(Runnable):
    def __init__(self):
        pass

    def invoke(self, input_data: dict) -> str:
        return input_data["content"]



