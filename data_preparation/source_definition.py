from abc import ABC, abstractmethod

from data_preparation.default_behaviours import MissingBehaviour


class SourceDefinition(ABC):
    """
    Abstract base class that defines the structure of a configuration of source
    """

    @property
    @abstractmethod
    def table_path(self) -> str:
        """
        Returns the path to the Python script that needs to be executed.
        """
        pass

    @property
    @abstractmethod
    def missing_behaviour(self) -> MissingBehaviour:
        """
        Returns the path to the Python script that needs to be executed.
        """
        pass

    @property
    @abstractmethod
    def default_missing(self):
        """
        Returns the path to the Python script that needs to be executed.
        """
        pass

    @property
    @abstractmethod
    def date_column(self):
        """
        Returns the path to the Python script that needs to be executed.
        """
        pass