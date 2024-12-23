from abc import ABC,abstractmethod

class TrainingModel(ABC):


    @property
    @abstractmethod
    def model_name(self):
        pass

    @property
    @abstractmethod
    def model_definition(self):
        pass

