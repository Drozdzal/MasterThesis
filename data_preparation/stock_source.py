from common.config import DATA_PATH
from data_preparation.default_behaviours import MissingBehaviour
from data_preparation.source_definition import SourceDefinition


class StockSource(SourceDefinition):
    @property
    def table_path(self) -> str:
        return f"../{DATA_PATH}/stock_gold.csv"

    @property
    def missing_behaviour(self) -> MissingBehaviour:
        return MissingBehaviour.DROP

    @property
    def default_missing(self):
        pass

    @property
    def date_column(self):
        return "Date"
