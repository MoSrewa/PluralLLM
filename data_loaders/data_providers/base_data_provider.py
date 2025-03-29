from abc import ABC, abstractmethod

class DataProvider(ABC):
    def __init__(self):
        """Empty constructor"""
        pass

    @abstractmethod
    def load_data_set(self):
        """
        Abstract method to return a list of clients.
        This must be implemented by any subclass.
        """
        pass
