from abc import ABC, abstractmethod


class Runnable(ABC):
    @abstractmethod
    def invoke(self, *args, **kwargs):
        pass


class RunnableConnector(Runnable):
    def __init__(self, runnable_list: list):
        self.runnable_list = runnable_list

    def invoke(self, input_data):
        data = input_data
        for runnable in self.runnable_list:
            data = runnable.invoke(data)
        return data


