from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

code = """
class Calculator:
    def add(self, a, b):
        return a + b

    def multiply(self, a, b):
        return a * b


class Statistics:
    def mean(self, values):
        return sum(values) / len(values)

    def max_value(self, values):
        return max(values)
"""


splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=200,
    chunk_overlap=20
)

chunks = splitter.split_text(code)

print(chunks[0])

# output:
# class Calculator:
#     def add(self, a, b):
#         return a + b
#
#     def multiply(self, a, b):
#         return a * b
